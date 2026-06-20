let chatBoxContainer;
let chatContainer;
let messageInput;
let chatId;
let replyToId = null;

document.addEventListener("DOMContentLoaded", () => {
    /**
     * Инициализация чата после полной загрузки DOM.
     *
     * Объявляет и инициализирует основные переменные интерфейса чата.
     * Загружает сообщения выбранного чата и организует автообновление.
     *
     * Args:
     *   None
     *
     * Returns:
     *   None
     */
    chatBoxContainer = document.getElementById("chatBox");
    chatContainer = document.getElementById("chat");
    messageInput = document.getElementById("messageInput");
    chatId = chatBoxContainer ? chatBoxContainer.dataset.chatId : null;
    if (chatId === "0") chatId = null;
    if (chatId) loadChatMessages(chatId);
    setInterval(() => {
        if (chatId) loadChatMessages(chatId);
    }, 4000);
});

/**
 * Проверяет, находится ли пользователь внизу сообщения чата.
 *
 * Args:
 *   None
 *
 * Returns:
 *   {boolean} True, если пользователь прокрутил чат к низу, иначе False.
 */
function isUserAtBottom() {
    const threshold = 150;
    return (chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight) < threshold;
}

/**
 * Показывает блок ответа для выбранного сообщения.
 *
 * Args:
 *   id {number|string} - id сообщения, на которое отвечают
 *   text {string} - текст сообщения, на которое отвечают
 *
 * Returns:
 *   None
 */
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

/**
 * Скрывает блок ответа и очищает replyToId.
 *
 * Args:
 *   None
 *
 * Returns:
 *   None
 */
function cancelReply() {
    replyToId = null;
    const replyBlock = document.getElementById("replyBlock");
    if (replyBlock) replyBlock.style.display = "none";
}

/**
 * Загружает все сообщения для чата по id.
 *
 * Args:
 *   id {number|string} - идентификатор чата
 *
 * Returns:
 *   None (Асинхронная функция, меняет DOM)
 */
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
            if (msg.reply_to_id && msg.reply_to_content) {
                const quoteDiv = document.createElement("div");
                quoteDiv.className = "reply-quote";
                quoteDiv.textContent = msg.reply_to_content;
                msgDiv.appendChild(quoteDiv);
            }
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
            const rBtn = document.createElement("button");
            rBtn.innerHTML = "↩";
            rBtn.className = "btn-reply-action";
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

/**
 * Отправляет текстовое сообщение в текущий чат.
 *
 * Args:
 *   None
 *
 * Returns:
 *   None (Асинхронная функция)
 */
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

/**
 * Загружает файл в чат. Автоматически определяет тип контента (файл/изображение).
 *
 * Args:
 *   None
 *
 * Returns:
 *   None (Асинхронная функция)
 */
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

/**
 * Обновляет выбранный чат в списке и загружает его сообщения.
 *
 * Args:
 *   newChatId {number|string} - id нового выбранного чата
 *   element {HTMLElement} - DOM элемент выбранного чата (для визуального выделения)
 *
 * Returns:
 *   None
 */
function updateSelectedChat(newChatId, element) {
    chatId = newChatId;
    document.querySelectorAll(".chat-item-styled").forEach(item => item.classList.remove("active"));
    if (element) element.classList.add("active");
    const name = element ? element.querySelector('.chat-name').textContent : "Чат";
    const header = document.getElementById('chatHeaderTitle');
    if (header) header.textContent = name;
    loadChatMessages(chatId);
}
