// BrainScan AI — app.js

document.addEventListener("DOMContentLoaded", () => {

  // ── File Drop Zone ──────────────────────────────────────────────────────
  const dropZone   = document.getElementById("dropZone");
  const fileInput  = document.getElementById("fileInput");
  const dropContent = document.getElementById("dropContent");
  const previewImg = document.getElementById("previewImg");
  const fileInfo   = document.getElementById("fileInfo");
  const fileName   = document.getElementById("fileName");
  const predictForm = document.getElementById("predictForm");
  const loadingOverlay = document.getElementById("loadingOverlay");

  if (!dropZone) return;

  // Click on zone → trigger file input
  dropZone.addEventListener("click", (e) => {
    if (e.target !== fileInput) fileInput.click();
  });

  // Drag events
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFile(e.dataTransfer.files[0]);
    }
  });

  // File input change
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file.");
      return;
    }
    // Show file name bar
    fileName.textContent = file.name;
    fileInfo.classList.remove("hidden");

    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewImg.classList.remove("hidden");
      dropContent.classList.add("hidden");
    };
    reader.readAsDataURL(file);
  }

  // Form submit → show loading overlay
  if (predictForm) {
    predictForm.addEventListener("submit", () => {
      if (fileInput.files.length > 0) {
        loadingOverlay.classList.remove("hidden");
      }
    });
  }
});

// Remove selected file
function removeFile() {
  const fileInput  = document.getElementById("fileInput");
  const dropContent = document.getElementById("dropContent");
  const previewImg = document.getElementById("previewImg");
  const fileInfo   = document.getElementById("fileInfo");
  const fileName   = document.getElementById("fileName");

  fileInput.value  = "";
  previewImg.src   = "";
  previewImg.classList.add("hidden");
  dropContent.classList.remove("hidden");
  fileInfo.classList.add("hidden");
  fileName.textContent = "No file chosen";
}

// Auto-dismiss flash messages after 4 seconds
document.querySelectorAll(".alert").forEach((el) => {
  setTimeout(() => {
    el.style.transition = "opacity .4s";
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 400);
  }, 4000);
});

// Animate probability bars on result page
document.querySelectorAll(".prob-bar").forEach((bar) => {
  const w = bar.style.width;
  bar.style.width = "0%";
  setTimeout(() => { bar.style.width = w; }, 100);
});
