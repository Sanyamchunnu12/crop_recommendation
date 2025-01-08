// Navbar Toggle for Mobile
const menuToggle = document.querySelector('.menu-toggle');
const navbar = document.querySelector('.navbar');

menuToggle.addEventListener('click', () => {
    navbar.classList.toggle('active');
});

// Automatic Slider
const slides = document.querySelector('.slides');
const slideImages = document.querySelectorAll('.slide');
let currentIndex = 0;
const slideInterval = 5000; 

function nextSlide() {
    currentIndex = (currentIndex + 1) % slideImages.length;
    updateSlider();
}

function updateSlider() {
    slides.style.transform = `translateX(-${currentIndex * 100}%)`;
}

// Start the slider
setInterval(nextSlide, slideInterval);
