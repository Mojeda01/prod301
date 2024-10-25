document.addEventListener("DOMContentLoaded", function() {
    const infoIcon = document.querySelector(".info-icon i");
    const popup = document.querySelector(".info-popup");
    const closeBtn = document.querySelector(".info-popup .close-btn");

    infoIcon.addEventListener("click", function() {
        popup.style.display = "block";
    });

    closeBtn.addEventListener("click", function() {
        popup.style.display = "none";
    });
});