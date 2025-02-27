<template>
  <div class="comments-section">
    <h3>Comments</h3>
    
    <!-- Add comment form -->
    <form @submit.prevent="addComment" class="comment-form">
      <div class="form-group">
        <input 
          v-model="newComment.name" 
          type="text" 
          placeholder="Your Name" 
          required
          :disabled="isLoading"
        >
      </div>
      <div class="form-group">
        <textarea 
          v-model="newComment.content" 
          placeholder="Write your comment..." 
          required
          :disabled="isLoading"
        ></textarea>
      </div>
      <button type="submit" :disabled="isLoading">
        {{ isLoading ? 'Posting...' : 'Post Comment' }}
      </button>
    </form>

    <!-- Comments list -->
    <div class="comments-list">
      <div v-if="error" class="error-message">
        {{ error }}
      </div>
      <div v-else-if="isLoading" class="loading">
        Loading comments...
      </div>
      <div v-else-if="comments.length === 0" class="no-comments">
        Be the first to comment!
      </div>
      <div v-else v-for="comment in comments" :key="comment.id" class="comment">
        <div class="comment-header">
          <strong>{{ comment.name }}</strong>
          <span>{{ formatDate(comment.date) }}</span>
        </div>
        <p>{{ comment.content }}</p>
      </div>
    </div>
  </div>
</template>

<script>
import { JSONBIN_CONFIG } from '@/config/jsonbin';

export default {
  name: 'BlogComments',
  props: {
    blogId: {
      type: String,
      required: true
    }
  },
  data() {
    return {
      comments: [],
      newComment: {
        name: '',
        content: ''
      },
      isLoading: false,
      error: null
    }
  },
  methods: {
    formatDate(date) {
      return new Date(date).toLocaleDateString();
    },
    async fetchComments() {
      this.isLoading = true;
      this.error = null;
      try {
        const response = await fetch(`${process.env.VUE_APP_JSONBIN_BASE_URL}/${process.env.VUE_APP_JSONBIN_BIN_ID}`, {
          headers: {
            'X-Access-Key': process.env.VUE_APP_JSONBIN_API_KEY
          }
        });
        
        if (!response.ok) throw new Error('Failed to fetch comments');
        
        const data = await response.json();
        const allComments = data.record.blog_comments || {};
        this.comments = allComments[this.blogId] || [];
      } catch (error) {
        console.error('Error fetching comments:', error);
        this.error = 'Failed to load comments. Please try again later.';
      } finally {
        this.isLoading = false;
      }
    },
    async addComment() {
      this.isLoading = true;
      this.error = null;
      try {
        const response = await fetch(`${JSONBIN_CONFIG.BASE_URL}/${JSONBIN_CONFIG.BIN_ID}`, {
          headers: {
            'X-Master-Key': JSONBIN_CONFIG.API_KEY
          }
        });
        
        if (!response.ok) throw new Error('Failed to fetch current comments');
        
        const data = await response.json();
        const allComments = data.record.blog_comments || {};
        
        // Prepare new comment
        const newComment = {
          id: Date.now().toString(),
          ...this.newComment,
          date: new Date().toISOString()
        };

        // Initialize blog comments array if it doesn't exist
        if (!allComments[this.blogId]) {
          allComments[this.blogId] = [];
        }
        allComments[this.blogId].push(newComment);

        // Update JSONbin with new structure
        await this.saveComments({ blog_comments: allComments });

        // Update local state
        this.comments = allComments[this.blogId];
        
        // Reset form
        this.newComment = {
          name: '',
          content: ''
        };
      } catch (error) {
        console.error('Error adding comment:', error);
        this.error = 'Failed to add comment. Please try again later.';
      } finally {
        this.isLoading = false;
      }
    },
    async saveComments(comments) {
      try {
        await fetch(`${process.env.VUE_APP_JSONBIN_BASE_URL}/${process.env.VUE_APP_JSONBIN_BIN_ID}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
            'X-Access-Key': process.env.VUE_APP_JSONBIN_API_KEY
          },
          body: JSON.stringify(comments)
        });
      } catch (error) {
        console.error('Error saving comments:', error);
        this.error = 'Failed to save comments. Please try again later.';
      }
    }
  },
  mounted() {
    this.fetchComments();
  }
}
</script>

<style scoped>
.comments-section {
  margin-top: 3rem;
  padding-top: 2rem;
  border-top: 1px solid #eee;
}

.comment-form {
  margin-bottom: 2rem;
}

.form-group {
  margin-bottom: 1rem;
}

.form-group input,
.form-group textarea {
  width: 100%;
  padding: 0.8rem;
  border: 1px solid #ddd;
  border-radius: 8px;
  font-size: 1rem;
  background: #f8f9fa;
  transition: all 0.3s ease;
}

.form-group input:focus,
.form-group textarea:focus {
  outline: none;
  border-color: #3498db;
  background: #fff;
}

.form-group textarea {
  min-height: 100px;
  resize: vertical;
}

button {
  padding: 0.8rem 1.5rem;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

button:hover {
  background: #2980b9;
  transform: translateY(-2px);
}

button:disabled {
  background: #ccc;
  cursor: not-allowed;
  transform: none;
}

.comments-list {
  margin-top: 2rem;
}

.loading,
.error-message,
.no-comments {
  padding: 2rem;
  text-align: center;
  color: #666;
  font-style: italic;
}

.comment {
  background: white;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
  margin-bottom: 1rem;
}

.comment-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
  color: #666;
}

/* Add responsive styles for code blocks */
@media screen and (max-width: 768px) {
  pre {
    max-width: 100%;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  code {
    font-size: 0.9em;
  }
}
</style>
