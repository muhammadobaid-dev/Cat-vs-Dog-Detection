const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const preview = document.getElementById("preview");
const resultPanel = document.getElementById("resultPanel");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");
const resultEyebrow = document.getElementById("resultEyebrow");
const resultMeta = document.getElementById("resultMeta");
const catPct = document.getElementById("catPct");
const dogPct = document.getElementById("dogPct");
const catBar = document.getElementById("catBar");
const dogBar = document.getElementById("dogBar");
const statusEl = document.getElementById("status");
const yearEl = document.getElementById("year");

yearEl.textContent = String(new Date().getFullYear());

function setStatus(message, type = "") {
  statusEl.textContent = message;
  statusEl.className = `status${type ? ` is-${type}` : ""}`;
}

function showPreview(file) {
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.hidden = false;
  dropzone.classList.add("has-image");
}

function renderResult(data) {
  resultPanel.hidden = false;
  resultEyebrow.textContent = "Muhammad Ubaid · Prediction";
  resultLabel.textContent = data.label;
  resultConfidence.textContent = `${data.confidence}%`;
  catPct.textContent = `${data.scores.cat}%`;
  dogPct.textContent = `${data.scores.dog}%`;
  requestAnimationFrame(() => {
    catBar.style.width = `${data.scores.cat}%`;
    dogBar.style.width = `${data.scores.dog}%`;
  });
  resultMeta.textContent = `Engine: ${data.engine}`;
}

async function predict(file) {
  setStatus("Analyzing image…", "loading");
  resultPanel.hidden = true;

  const body = new FormData();
  body.append("file", file);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      body,
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Prediction failed");
    }
    renderResult(data);
    setStatus("Done — ready for another photo.");
  } catch (err) {
    setStatus(err.message || "Something went wrong.", "error");
  }
}

async function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("Please choose an image file.", "error");
    return;
  }
  showPreview(file);
  await predict(file);
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) handleFile(file);
});

["dragenter", "dragover"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("is-dragover");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
  });
});

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFile(file);
});
