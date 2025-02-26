require('dotenv').config();
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const { OpenAI } = require('openai');

class LocalLLMService {
  constructor() {
    const projectRoot = path.resolve(__dirname, '../../');
    this.modelPath = path.join(projectRoot, 'models/llama-2-7b-chat.gguf');
    this.llamaPath = '/usr/local/Cellar/llama.cpp/4778/bin/llama-cli'; // Updated to use Homebrew path
    
    // Debug logging
    console.log('Paths:', {
      projectRoot,
      modelPath: this.modelPath,
      llamaPath: this.llamaPath,
      exists: {
        model: fs.existsSync(this.modelPath),
        llama: fs.existsSync(this.llamaPath)
      },
      permissions: this.getFilePermissions(this.llamaPath)
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
}

module.exports = new LocalLLMService();
