const form = document.querySelector("#upload-form");
const fileInput = document.querySelector("#file-input");
const dropZone = document.querySelector("#drop-zone");
const selectedFile = document.querySelector("#selected-file");
const fileName = document.querySelector("#file-name");
const submitButton = document.querySelector("#submit-button");
const statusMessage = document.querySelector("#status-message");
const resultPanel = document.querySelector("#result-panel");
const preview = document.querySelector("#image-preview");

let selectedImage = null;
let latestResult = null;
let previewUrl = null;

function chooseFile(file) {
  if (!file) return;
  const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
  if (!allowedTypes.includes(file.type)) {
    showStatus("Choose a JPEG, PNG, or WebP image.", false);
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showStatus("This image is larger than the 10 MB limit.", false);
    return;
  }

  selectedImage = file;
  fileName.textContent = file.name;
  selectedFile.hidden = false;
  submitButton.disabled = false;
  dropZone.classList.add("has-preview");
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  showStatus("", false);
  resultPanel.hidden = true;
}

function clearFile() {
  selectedImage = null;
  fileInput.value = "";
  selectedFile.hidden = true;
  submitButton.disabled = true;
  dropZone.classList.remove("has-preview");
  preview.hidden = true;
  preview.removeAttribute("src");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
}

function showStatus(message, success) {
  statusMessage.textContent = message;
  statusMessage.hidden = !message;
  statusMessage.classList.toggle("success", success);
}

function setLoading(loading) {
  submitButton.disabled = loading || !selectedImage;
  document.querySelector("#button-label").textContent = loading ? "Processing…" : "Extract details";
  document.querySelector("#button-arrow").hidden = loading;
  document.querySelector("#button-spinner").hidden = !loading;
}

function renderResults(result) {
  latestResult = result;
  const labels = [
    ["Full name", result.extracted_data.full_name],
    ["Document number", result.extracted_data.document_number],
    ["Date of birth", result.extracted_data.date_of_birth],
    ["Expiry date", result.extracted_data.expiry_date],
  ];
  const fields = document.querySelector("#fields-grid");
  fields.replaceChildren();
  labels.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "field-card";
    const labelNode = document.createElement("span");
    labelNode.className = "field-label";
    labelNode.textContent = label;
    const valueNode = document.createElement("span");
    valueNode.className = `field-value${value ? "" : " missing"}`;
    valueNode.textContent = value || "Not detected";
    card.append(labelNode, valueNode);
    fields.append(card);
  });

  const portrait = document.querySelector("#portrait-image");
  portrait.src = `${result.image_metadata.face_thumbnail_url}?record=${encodeURIComponent(result.id)}`;
  document.querySelector("#face-status").textContent = result.image_metadata.face_detected
    ? "Face detected"
    : "Portrait-area crop · face not confirmed";
  document.querySelector("#raw-text").textContent = result.raw_text || "No text was recognized.";
  document.querySelector("#record-id").textContent =
    `Saved record · ${result.id} · ${result.image_metadata.width} × ${result.image_metadata.height}px`;
  resultPanel.hidden = false;
  resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

dropZone.addEventListener("click", (event) => {
  if (event.target.closest("#image-preview")) return;
  fileInput.click();
});
fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));
document.querySelector("#remove-file").addEventListener("click", clearFile);

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
});
dropZone.addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedImage) return;
  const payload = new FormData();
  payload.append("file", selectedImage);
  setLoading(true);
  showStatus("Detecting the card and reading its details…", true);
  try {
    const response = await fetch("/api/v1/process-id", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok) {
      const message = typeof data.detail === "string" ? data.detail : "The image could not be processed.";
      throw new Error(message);
    }
    renderResults(data);
    showStatus("Extraction complete. Review the fields for accuracy.", true);
  } catch (error) {
    showStatus(error.message || "Could not connect to the ID Lens API.", false);
  } finally {
    setLoading(false);
  }
});

document.querySelector("#copy-json").addEventListener("click", async (event) => {
  if (!latestResult) return;
  const button = event.currentTarget;
  try {
    await navigator.clipboard.writeText(JSON.stringify(latestResult, null, 2));
    button.textContent = "Copied!";
    window.setTimeout(() => { button.textContent = "Copy JSON"; }, 1500);
  } catch {
    showStatus("Clipboard access was blocked by your browser.", false);
  }
});
