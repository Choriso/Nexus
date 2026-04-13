// --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
let chatBoxContainer;
let chatContainer;
let messageInput;
let chatId;
let replyToId = null;

// --- ИНИЦИАЛИЗАЦИЯ ---
document.addEventListener("DOMContentLoaded", () => {
    // Инициализируем переменные, когда DOM уже готов
    chatBoxContainer = document.getElementById("chatBox");
    chatContainer = document.getElementById("chat");
    messageInput = document.getElementById("messageInput");

    chatId = chatBoxContainer ? chatBoxContainer.dataset.chatId : null;
    if (chatId === "0") chatId = null;

    if (chatId) loadChatMessages(chatId);

    // Автообновление
    setInterval(() => {
        if (chatId) loadChatMessages(chatId);
    }, 4000);
});

// Далее все остальные функции (isUserAtBottom, sendMessage и т.д.) без изменений


function isUserAtBottom() {
    const threshold = 150;
    return (chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight) < threshold;
}

// Показать блок ответа
function replyToMessage(id, text) {
    replyToId = id;
    const replyBlock = document.getElementById("replyBlock");
    const replyContent = document.getElementById("replyContent");

    if (replyBlock && replyContent) {
        replyContent.textContent = text;
        replyBlock.style.display = "flex";
        messageInput.focus();
    }
}

// Отмена ответа
function cancelReply() {
    replyToId = null;
    const replyBlock = document.getElementById("replyBlock");
    if (replyBlock) replyBlock.style.display = "none";
}

// --- ОСНОВНЫЕ ФУНКЦИИ ЧАТА ---

async function loadChatMessages(id) {
    if (!id || !chatContainer) return;

    const wasAtBottom = isUserAtBottom();

    try {
        const response = await fetch(`/chat/messages/${id}`);
        const data = await response.json();

        chatContainer.innerHTML = "";
        const messages = data.messages || data;

        if (messages.length === 0) {
            chatContainer.innerHTML = '<div class="chat-placeholder">Сообщений пока нет</div>';
        }

        messages.forEach(msg => {
            const msgDiv = document.createElement("div");
            msgDiv.className = `message ${msg.sent_by_user ? 'sent' : 'received'}`;

            // Рендер цитаты ответа
            if (msg.reply_to_id && msg.reply_to_content) {
                const quoteDiv = document.createElement("div");
                quoteDiv.className = "reply-quote";
                quoteDiv.textContent = msg.reply_to_content;
                msgDiv.appendChild(quoteDiv);
            }

            // Рендер основного контента
            if (msg.message_type === 'image') {
                const img = document.createElement('img');
                img.src = msg.content;
                img.onclick = () => window.open(msg.content, '_blank');
                msgDiv.appendChild(img);
            } else if (msg.message_type === 'file') {
                const fileLink = document.createElement("a");
                fileLink.href = msg.content;
                fileLink.className = "file-link";
                fileLink.textContent = "📂 Файл";
                fileLink.target = "_blank";
                msgDiv.appendChild(fileLink);
            } else {
                const textSpan = document.createElement("span");
                textSpan.textContent = msg.content || msg.text;
                msgDiv.appendChild(textSpan);
            }

            // Кнопка ответа
            const rBtn = document.createElement("button");
            rBtn.innerHTML = "↩";
            rBtn.className = "btn-reply-action"; // добавь стиль в CSS
            rBtn.onclick = () => replyToMessage(msg.id, msg.content || msg.text);
            msgDiv.appendChild(rBtn);

            chatContainer.appendChild(msgDiv);
        });

        if (wasAtBottom) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    } catch (err) {
        console.error("Ошибка загрузки:", err);
    }
}

async function sendMessage() {
    const content = messageInput.value.trim();
    if (!content || !chatId) return;

    const body = {
        content: content,
        chat_id: parseInt(chatId),
        type: "text",
        reply_to_id: replyToId
    };

    try {
        const res = await fetch("/messages", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });
        const data = await res.json();

        if (data.status === "ok") {
            messageInput.value = "";
            cancelReply();
            loadChatMessages(chatId);
        }
    } catch (err) {
        console.error("Ошибка отправки:", err);
    }
}

async function uploadFile() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    if (!file || !chatId) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/upload", { method: "POST", body: formData });
        const result = await response.json();

        if (result.file_url) {
            const isImg = /\.(jpg|jpeg|png|webp|gif)$/i.test(file.name);
            await fetch("/messages", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    content: result.file_url,
                    chat_id: parseInt(chatId),
                    type: isImg ? 'image' : 'file',
                    reply_to_id: replyToId
                })
            });
            loadChatMessages(chatId);
            cancelReply();
        }
        fileInput.value = "";
    } catch (err) {
        console.error("Ошибка загрузки файла:", err);
    }
}

function updateSelectedChat(newChatId, element) {
    chatId = newChatId;
    document.querySelectorAll(".chat-item-styled").forEach(item => item.classList.remove("active"));
    if (element) element.classList.add("active");

    const name = element ? element.querySelector('.chat-name').textContent : "Чат";
    const header = document.getElementById('chatHeaderTitle');
    if (header) header.textContent = name;

    loadChatMessages(chatId);
}

