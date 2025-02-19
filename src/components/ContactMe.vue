<template>
  <article class="contact-form">
    <h2>Submit a request</h2>
    <form @submit.prevent="handleSubmit">
      <div class="form-group">
        <label for="email">Your email address</label>
        <input id="email" v-model="form.email" type="email" required />
      </div>
      <div class="form-group">
        <label for="subject">Subject</label>
        <input id="subject" v-model="form.subject" type="text" required />
      </div>
      <div class="form-group">
        <label for="description">Description</label>
        <textarea id="description" v-model="form.description" required placeholder="Enter your detailed question or description..."></textarea>
      </div>
      <div class="form-group">
        <label for="attachments">Attachments</label>
        <input id="attachments" type="file" @change="handleFileChange" multiple />
      </div>
      <button type="submit">Submit</button>
    </form>
  </article>
</template>

<script>
export default {
  name: "ContactMe",
  data() {
    return {
      form: {
        email: "",
        subject: "",
        description: ""
      },
      attachments: [] // store selected files
    };
  },
  methods: {
    handleFileChange(e) {
      // Capture all selected files in attachments array
      this.attachments = Array.from(e.target.files);
    },
    handleSubmit() {
      console.log("Form submitted", this.form, this.attachments);
      alert("Thank you for contacting me!");
      // Reset form and attachments
      this.form.email = "";
      this.form.subject = "";
      this.form.description = "";
      this.attachments = [];
      // Optionally clear file input here if needed using refs
    }
  }
}
</script>

<style scoped>
.contact-form {
  max-width: 1080px;
  margin: 3rem auto;       /* center horizontally */
  padding: 2rem;
  font-size: 18px;
  line-height: 1.8;
  font-family: 'Georgia', serif;
  color: #1a1a1a;
  background-color: transparent; /* remove box background */
  border-radius: 0;              /* remove rounded corners */
  box-shadow: none;              /* remove shadow */
  border: none;                  /* remove any border */
  text-align: center;
}

.contact-form h2 {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  font-family: 'Merriweather', serif;
  font-weight: 700;
  line-height: 1.2;
}

.form-group {
  margin-bottom: 2rem;
}

.contact-form label {
  display: block;
  font-weight: bold;
  margin-bottom: 0.5rem;
}

.contact-form input[type="text"],
.contact-form input[type="email"],
.contact-form textarea,
.contact-form input[type="file"] {
  width: 100%;
  padding: 0.75rem 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 1rem;
  box-sizing: border-box;
  background-color: #f9f9f9;
  transition: background-color 0.3s ease, border-color 0.3s ease;
}

.contact-form textarea {
  min-height: 150px;
  resize: vertical;
}

.contact-form input:focus,
.contact-form textarea:focus {
  background-color: #fff;
  border-color: #ccc;
  outline: none;
}

.contact-form button {
  display: block;
  width: 100%;
  padding: 0.75rem;
  background-color: #1a8917;
  color: #fff;
  border: none;
  border-radius: 4px;
  font-size: 1.1rem;
  cursor: pointer;
  transition: background-color 0.3s ease;
  margin-top: 1rem;
}

.contact-form button:hover {
  background-color: #137c13;
}
</style>
