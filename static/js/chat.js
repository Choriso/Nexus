class Message {
    constructor(id, author, content, type = "text", timestamp = new Date().toLocaleTimeString()) {
        this.id = id;
        this.author = author;
        this.content = content;
        this.type = type;
        this.timestamp = timestamp;
    }

    render() {
        let messageElement = document.createElement("div");
        messageElement.classList.add("message");
        messageElement.classList.add(this.author === "Вы" ? "sent" : "received");

        let timeElement = document.createElement("span");
        timeElement.classList.add("timestamp");
        timeElement.innerText = this.timestamp;

        let authorElement = document.createElement("strong");
        authorElement.innerText = this.author + ": ";

        let contentElement = document.createElement("span");

        let buttonsContainer = document.createElement("div");
        buttonsContainer.classList.add("message-buttons");

        let deleteButton = document.createElement("button");
        deleteButton.innerText = "🗑";
        deleteButton.onclick = () => deleteMessage(this.id);

        let replyButton = document.createElement("button");
        replyButton.innerText = "💬";
        replyButton.onclick = () => replyMessage(this.content);

        if (this.type === "text") {
            contentElement.innerText = this.content;
            let editButton = document.createElement("button");
            editButton.innerText = "✏";
            editButton.onclick = () => editMessage(this.id, this.content);
            buttonsContainer.append(replyButton, editButton, deleteButton);
        } else if (this.type === "image") {
            let img = document.createElement("img");
            img.src = this.content;
            img.classList.add("chat-image");
            img.style.maxWidth = "150px";
            img.style.cursor = "pointer";
            contentElement.appendChild(img);

            let fileLink = document.createElement("a");
            fileLink.href = this.content;
            fileLink.innerText = "📂";
            fileLink.download = "";
            buttonsContainer.append(fileLink, replyButton, deleteButton);
        } else if (this.type === "file") {
            let fileLink = document.createElement("a");
            fileLink.href = this.content;
            fileLink.innerText = "📂" + this.content.split("/").pop();
            fileLink.download = "";
            contentElement.appendChild(fileLink);
            buttonsContainer.append(replyButton, deleteButton);
        }

        messageElement.append(authorElement, contentElement, timeElement, buttonsContainer);
        return messageElement;
    }
}

const chatContainer = document.getElementById("chat");
const chatBox = document.querySelector(".chat-box");
let chatId = chatBox ? chatBox.getAttribute("data-chat-id") : null;

function replyMessage(content) {
    let input = document.getElementById("messageInput");
    input.value = `> ${content}\n`;
    input.focus();
}

async function editMessage(id, oldContent) {
    let newContent = prompt("Измените сообщение:", oldContent);
    if (!newContent || newContent === oldContent) return;

    await fetch(`/messages/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: newContent })
    });
    loadMessages(chatId);
}

async function deleteMessage(id) {
    if (!confirm("Удалить сообщение?")) return;
    await fetch(`/messages/${id}`, { method: "DELETE" });
    loadMessages(chatId);
}

async function loadMessages(id) {
    if (!id) {
        console.error("Chat ID отсутствует!");
        return;
    }

    try {
        let response = await fetch(`/chat/messages/${id}`);
        if (!response.ok) throw new Error("Ошибка загрузки сообщений");

        let data = await response.json();
        chatContainer.innerHTML = "";
        data.messages.forEach(msg => {
            let message = new Message(msg.id, msg.sent_by_user ? "Вы" : data.chat_name, msg.text, msg.message_type, msg.timestamp);
            chatContainer.appendChild(message.render());
        });

        // Обновляем заголовок чата
        const chatHeader = document.querySelector(".chat-header");
        if (chatHeader) chatHeader.textContent = data.chat_name;
    } catch (err) {
        console.error(err);
    }
}

async function sendMessage() {
    let input = document.getElementById("messageInput");
    let content = input.value.trim();
    if (!content || !chatId) return;

    await fetch("/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content, chat_id: chatId, type: "text" })
    });

    input.value = "";
    loadMessages(chatId);
}

async function uploadFile() {
    let fileInput = document.getElementById("fileInput");
    let file = fileInput.files[0];
    if (!file || !chatId) return;

    let formData = new FormData();
    formData.append("file", file);

    let response = await fetch("/upload", { method: "POST", body: formData });
    let result = await response.json();

    if (result.file_url) {
        let type = result.file_url.match(/\.(jpg|jpeg|png|gif)$/i) ? "image" : "file";
        await fetch("/messages", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content: result.file_url, type, chat_id: chatId })
        });
        loadMessages(chatId);
    }
}

if (chatId) loadMessages(chatId);
setInterval(() => loadMessages(chatId), 3000);
