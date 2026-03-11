// Wrap each nbsphinx code input cell in a collapsible <details> element.
document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".nbinput .input_area").forEach(function (inputArea) {
        var details = document.createElement("details");
        var summary = document.createElement("summary");
        summary.textContent = "Show code";
        details.appendChild(summary);

        // Move all children of inputArea into <details>
        while (inputArea.firstChild) {
            details.appendChild(inputArea.firstChild);
        }
        inputArea.appendChild(details);
    });
});
