<template>
  <div class="chatbot-widget">
    <button class="chatbot-toggler" @click="toggleChatbot">
      <span v-if="!isOpen">K8s Chat</span>
      <span v-else>Close</span>
    </button>
    <div v-if="isOpen" class="chatbot-window">
      <div class="chat-header">
        <h2>Kubernetes Chatbot</h2>
      </div>
      <div class="chat-history" ref="chatHistory">
        <div v-for="(message, index) in messages" :key="index" :class="['message', message.sender]">
          <p>{{ message.text }}</p>
        </div>
      </div>
      <div class="chat-input">
        <input type="text" v-model="userInput" @keyup.enter="sendMessage" placeholder="Ask about Kubernetes..." />
        <button @click="sendMessage">Send</button>
      </div>
    </div>
  </div>
</template>

<script>
import LocalLLMService from '@/services/LocalLLMService';

export default {
  name: 'ChatbotWidget',
  data() {
    return {
      isOpen: false,
      userInput: '',
      messages: [
        { sender: 'bot', text: 'Hello! How can I help you with Kubernetes today?' }
      ]
    };
  },
  methods: {
    toggleChatbot() {
      this.isOpen = !this.isOpen;
    },
    sendMessage() {
      if (this.userInput.trim() === '') return;

      const userMessage = this.userInput;
      this.messages.push({ sender: 'user', text: userMessage });
      this.userInput = ''; // Clear input immediately

      // Call LLM Service
      LocalLLMService.getChatbotResponse(userMessage)
        .then(botResponse => {
          this.messages.push({ sender: 'bot', text: botResponse });
        })
        .catch(error => {
          console.error("Error sending message to LLM:", error);
          this.messages.push({ sender: 'bot', text: 'Sorry, I had trouble getting a response. Please try again.' });
        })
        .finally(() => {
          this.$nextTick(() => {
            this.scrollToBottom();
          });
        });
    },
    scrollToBottom() {
      const chatHistory = this.$refs.chatHistory;
      if (chatHistory) {
        chatHistory.scrollTop = chatHistory.scrollHeight;
      }
    }
  }
};
</script>

<style scoped>
.chatbot-widget {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 1000;
}

.chatbot-toggler {
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 50%;
  width: 60px;
  height: 60px;
  font-size: 1rem;
  cursor: pointer;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
}

.chatbot-window {
  width: 350px;
  height: 500px;
  background-color: white;
  border-radius: 10px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  margin-bottom: 10px; /* Space between window and toggler if window is above */
}

.chat-header {
  background-color: #007bff;
  color: white;
  padding: 15px;
  text-align: center;
}

.chat-header h2 {
  margin: 0;
  font-size: 1.2rem;
}

.chat-history {
  flex-grow: 1;
  padding: 15px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.message {
  padding: 10px;
  border-radius: 7px;
  max-width: 80%;
  word-wrap: break-word;
}

.message.user {
  background-color: #007bff;
  color: white;
  align-self: flex-end;
}

.message.bot {
  background-color: #f1f1f1;
  color: #333;
  align-self: flex-start;
}

.chat-input {
  display: flex;
  padding: 10px;
  border-top: 1px solid #eee;
}

.chat-input input {
  flex-grow: 1;
  border: 1px solid #ddd;
  border-radius: 5px;
  padding: 10px;
  margin-right: 10px;
}

.chat-input button {
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 5px;
  padding: 10px 15px;
  cursor: pointer;
}

.chat-input button:hover {
  background-color: #0056b3;
}
</style>
