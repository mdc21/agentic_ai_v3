let sessionId = localStorage.getItem('agent_session_id') || null;

const messagesContainer = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const resetBtn = document.getElementById('reset-btn');

function addMessage(text, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role} animate-in`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;

    messageDiv.appendChild(bubble);
    messagesContainer.appendChild(messageDiv);

    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.id = 'typing-indicator';
    indicator.className = 'message bot typing animate-in';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    messagesContainer.appendChild(indicator);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

async function sendMessage(text) {
    addMessage(text, 'user');
    userInput.value = '';

    showTypingIndicator();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                session_id: sessionId
            })
        });

        const data = await response.json();

        removeTypingIndicator();

        if (data.session_id) {
            sessionId = data.session_id;
            localStorage.setItem('agent_session_id', sessionId);
        }

        addMessage(data.response, 'bot');

    } catch (error) {
        removeTypingIndicator();
        addMessage('Sorry, I encountered an error. Please try again or reset the session.', 'bot');
        console.error('Error:', error);
    }
}

chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = userInput.value.trim();
    if (text) sendMessage(text);
});

resetBtn.addEventListener('click', async () => {
    if (!sessionId) return;

    if (confirm('Are you sure you want to reset the current session?')) {
        await fetch(`/reset?session_id=${sessionId}`, { method: 'POST' });
        sessionId = null;
        localStorage.removeItem('agent_session_id');
        messagesContainer.innerHTML = '';
        addMessage("Session reset. Hello! I'm your AI insurance assistant. How can I help you with your policy today?", 'bot');
    }
});
