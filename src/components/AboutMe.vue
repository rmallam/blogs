<template>
  <article class="about-me">
    <h2>About Me</h2>
    <p>
      Hi, I'm Rakesh Kumar Mallam, a Senior Architect at Red Hat. As a passionate technologist, I love exploring the ever-changing world of technology. Outside of my professional life, I enjoy cooking and playing tennis. I'm also a proud dad of two wonderful boys. Welcome to my blog where I share my insights and experiences in AI and ML.
      <br>email: mallamrakesh@gmail.com
      <br>linkedin: https://www.linkedin.com/in/rakesh-kumar-mallam-7b1b3b1/
    </p>
  </article>
</template>

<script>
export default {
  name: "AboutMe"
}
</script>

<style scoped>
.about-me {
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
.about-me h2 {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  font-family: 'Merriweather', serif;
  font-weight: 700;
  line-height: 1.2;
}
.about-me p {
  margin-bottom: 1.5rem;
}
</style>
