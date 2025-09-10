const chatForm = document.getElementById('chatForm');
const messageInput = chatForm.querySelector('textarea');
const chatHistory = document.getElementById('chatHistory');

messageInput.addEventListener('keydown', (e) => {
    // Check for Enter without Shift
    if (e.key === 'Enter' && !e.shiftKey){
        e.preventDefault();
        chatForm.requestSubmit();
    }
});

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

    // Update message count
    lastMessageCount = messages.length;

    // Re-render messages and context
    chatHistory.innerHTML = '';

    messages.forEach(msg => {
        // Initialize message div
        const message_div = document.createElement('div');
        // Set Class
        message_div.classList.add('message-box', msg.sender === 'human' ? 'you' : 'other');
        // Generate context links if machine user, otherwise blank string
        var links_section = ""
        if(msg.sender === 'machine'){
            var link_tags = ""
            for(let i=0; i<msg.context.length; i++){
                link_tags = link_tags +  `<a class="document-link" href="${msg.context[i].url}" target="_blank" rel="noopener noreferrer">${msg.context[i].title}</a>\n`
            }

            links_section = `
            <p class="document-list-title">Response Context</p>
            ${link_tags}
            `;

        }
        // Add message and message links to message div
        message_div.innerHTML = `
            <div class="message-content">
                ${msg.message}
                ${links_section}
            </div>
        `;
        // Add message div to page
        chatHistory.appendChild(message_div);
    });

    // Scroll to new messages
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Fetch messages on load and every 2 seconds
setInterval(fetchMessages, 2000);
fetchMessages();