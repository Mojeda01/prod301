// Toggle password visibility
const togglePassword = document.getElementById('togglePassword');
const passwordField = document.getElementById('password');

togglePassword.addEventListener('click', function() {
    console.log("Eye icon clicked!"); // Debugging line to check if the click is detected
    // Toggle the type attribute between password and text
    const type = passwordField.getAttribute('type') === 'password' ? 'text' : 'password';
    passwordField.setAttribute('type', type);
    // Toggle the eye icon
    this.classList.toggle('fa-eye-slash');
});