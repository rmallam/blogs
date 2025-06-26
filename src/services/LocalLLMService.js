require('dotenv').config();
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { OpenAI } = require('openai');

class LocalLLMService {
  constructor() {
    // Use environment variables with fallbacks for local development
    this.modelPath = process.env.LLAMA_MODEL_PATH || path.join(process.cwd(), 'models/llama-2-7b-chat.gguf');
    this.llamaPath = process.env.LLAMA_BINARY_PATH || '/usr/local/Cellar/llama.cpp/4778/bin/llama-cli';
    
    // Debug logging
    console.log('LLM Configuration:', {
      modelPath: this.modelPath,
      llamaPath: this.llamaPath,
      envVars: {
        LLAMA_MODEL_PATH: process.env.LLAMA_MODEL_PATH,
        LLAMA_BINARY_PATH: process.env.LLAMA_BINARY_PATH,
        PWD: process.cwd()
      },
      exists: {
        model: fs.existsSync(this.modelPath),
        llama: fs.existsSync(this.llamaPath)
      },
      permissions: {
        model: this.getFilePermissions(this.modelPath),
        llama: this.getFilePermissions(this.llamaPath)
      }
    });

    this.openai = new OpenAI({
      apiKey: process.env.VUE_APP_OPENAI_API_KEY
    });
  }

  getFilePermissions(filePath) {
    try {
      return fs.statSync(filePath).mode & parseInt('777', 8);
    } catch (error) {
      return 'File not accessible';
    }
  }

  async generateBlogPost(topic) {
    // Check if local LLM is available
    if (fs.existsSync(this.llamaPath)) {
      try {
        return await this.generateWithLocalLLM(topic);
      } catch (error) {
        console.log('Local LLM failed:', error.message);
        return await this.generateWithOpenAI(topic);
      }
    } else {
      console.log('Local LLM not found, using OpenAI');
      return await this.generateWithOpenAI(topic);
    }
  }

  async generateWithLocalLLM(topic) {
    const prompt = `Write a technical blog post about ${topic}. Include code examples where relevant. Format in markdown.`;
    
    console.log('Launching LLaMA with:', {
      binary: this.llamaPath,
      model: this.modelPath,
      exists: {
        binary: fs.existsSync(this.llamaPath),
        model: fs.existsSync(this.modelPath)
      }
    });

    return new Promise((resolve, reject) => {
      const llama = spawn(this.llamaPath, [
        '-m', this.modelPath,
        '-n', '2048',
        '--temp', '0.7',
        '--prompt', prompt
      ]);

      let output = '';
      let error = '';

      llama.stdout.on('data', (data) => {
        output += data.toString();
      });

      llama.stderr.on('data', (data) => {
        error += data.toString();
      });

      llama.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`LLaMA process failed: ${error}`));
          return;
        }

        const title = topic.toLowerCase().replace(/\s+/g, '_');
        const date = new Date().toISOString().split('T')[0];
        
        resolve({
          title: `${title}_${date}`,
          content: output,
          tags: ['llm_generated', topic.toLowerCase()]
        });
      });
    });
  }

  async generateWithOpenAI(topic) {
    const completion = await this.openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [
        {
          role: "system",
          content: "You are a technical blog writer specializing in cloud native technologies."
        },
        {
          role: "user",
          content: `Write a technical blog post about ${topic}. Include code examples where relevant. Format in markdown.`
        }
      ],
      temperature: 0.7,
      max_tokens: 2500
    });

    const blogContent = completion.choices[0].message.content;
    const title = topic.toLowerCase().replace(/\s+/g, '_');
    const date = new Date().toISOString().split('T')[0];
    
    return {
      title: `${title}_${date}`,
      content: blogContent,
      tags: ['ai_generated', topic.toLowerCase()]
    };
  }

  async getChatbotResponse(userPrompt) {
    // For now, we'll primarily use OpenAI for chat, as local LLM setup for chat is more complex
    // and the existing local LLM is geared towards blog post generation.
    try {
      const completion = await this.openai.chat.completions.create({
        model: "gpt-3.5-turbo", // Or any other preferred model
        messages: [
          {
            role: "system",
            content: "You are an expert AI assistant specializing in Kubernetes. Provide clear, concise, and accurate information about Kubernetes concepts, best practices, and troubleshooting."
          },
          {
            role: "user",
            content: userPrompt
          }
        ],
        temperature: 0.5, // Slightly lower temperature for more factual responses
        max_tokens: 500 // Adjust as needed
      });

      if (completion.choices && completion.choices.length > 0 && completion.choices[0].message) {
        return completion.choices[0].message.content.trim();
      } else {
        console.error('OpenAI response format error:', completion);
        return "Sorry, I couldn't get a response. Please try again.";
      }
    } catch (error) {
      console.error('Error getting chatbot response from OpenAI:', error);
      // Fallback or more specific error handling
      if (error.response && error.response.status === 401) {
        return "Error: Invalid OpenAI API key. Please check your configuration.";
      }
      if (error.code === 'ENOTFOUND') {
          return "Error: Could not connect to OpenAI. Please check your internet connection.";
      }
      return "Sorry, I encountered an error trying to respond. Please try again later.";
    }
  }
}

module.exports = new LocalLLMService();
