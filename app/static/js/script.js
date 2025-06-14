document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form");
  const textarea = document.querySelector("textarea");
  const button = document.querySelector("button");

  textarea.focus();

  const wordCountDisplay = document.createElement("p");
  wordCountDisplay.style.fontSize = "0.9em";
  wordCountDisplay.style.marginTop = "10px";
  wordCountDisplay.style.color = "#6c584c";
  textarea.parentNode.appendChild(wordCountDisplay);

  textarea.addEventListener("input", () => {
    const words = textarea.value.trim().split(/\s+/).filter(Boolean);
    wordCountDisplay.textContent = `📝 Word count: ${words.length}`;
  });

  form.addEventListener("submit", () => {
    button.disabled = true;
    button.textContent = "Analyzing...";
  });
});
