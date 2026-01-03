const chatWindow = document.getElementById("chatWindow");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");

function addMessage(text, type, extra = "") {
    const div = document.createElement("div");
    div.className = `message ${type} ${extra}`;
    div.textContent = text;
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return div;
}

async function sendMessage() {
    const query = userInput.value.trim();
    if (!query) return;

    userInput.value = "";
    sendBtn.disabled = true;

    addMessage(query, "user");

    const loading = addMessage("🤖 Thinking...", "bot", "loading");

    try {
        const res = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query })
        });

        if (!res.ok) throw new Error("API error");

        const data = await res.json();
        loading.remove();

        if (!data.answer) {
            addMessage("⚠️ No response received.", "bot", "error");
            return;
        }

        const botMsg = addMessage(data.answer, "bot");

        if (data.sources?.length) {
            const src = document.createElement("div");
            src.className = "sources";
            src.textContent = "Sources: " + data.sources.join(", ");
            botMsg.appendChild(src);
        }

    } catch {
        loading.remove();
        addMessage("❌ Unable to fetch response.", "bot", "error");
    } finally {
        sendBtn.disabled = false;
    }
}

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", e => {
    if (e.key === "Enter") sendMessage();
});
