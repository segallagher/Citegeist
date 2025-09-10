const chatForm = document.getElementById('chatForm')
const chatHistory = document.getElementById('chatHistory')

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    // Get message and username
    const message = chatForm.message.value.trim();
    if(!message) return;
    const username = chatForm.username.value;
    if(!username) return;

    // Empty Text field
    chatForm.message.value = '';

    // Send message
    await fetch('/send', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            username: username,
            message: message
        })
    });

    await fetchMessages();
})

let lastMessageCount = 0;

async function fetchMessages(){
    // Get Message list
    const res = await fetch('/messages');
    const messages = await res.json();

    // Only Update if new messages found
    if (messages.length === lastMessageCount){
        return;
    }

    console.log("Time to Update")
    // Update message count
    lastMessageCount = messages.length;

    // Re-render messages
    chatHistory.innerHTML = '';
    messages.forEach(msg => {
        const div = document.createElement('div');
        div.classList.add('message-box', msg.sender === 'human' ? 'you' : 'other');
        div.innerHTML = `<div class="message-content">${msg.message}</div>`;
        chatHistory.appendChild(div);
    });
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

setInterval(fetchMessages, 2000);
fetchMessages();