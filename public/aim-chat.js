const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const clearButton = document.getElementById('clear-button');
const saveButton = document.getElementById('save-button');
const backButton = document.getElementById('back-button');
const retryButton = document.getElementById('retry-button');
const conversationIdInput = document.getElementById('conversation-id');
const loadingIndicator = document.getElementById('loading-indicator');

const API_KEY = '123'; // Replace with your actual API key
const API_URL = '/v1/chat/completions';
const SAVE_URL = '/v1/save_conversation';

let conversationHistory = [];
let currentConversationId = null;

function generateConversationId() {
    return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function initializeConversation() {
    const savedData = localStorage.getItem('chatData');
    if (savedData) {
        const parsed = JSON.parse(savedData);
        conversationHistory = parsed.messages || [];
        currentConversationId = parsed.conversationId;
    }

    if (!currentConversationId) {
        currentConversationId = generateConversationId();
    }

    conversationIdInput.value = currentConversationId;
    return { conversationHistory, currentConversationId };
}

function loadConversationHistory() {
    const { conversationHistory } = initializeConversation();
    chatContainer.innerHTML = ''; // Clear existing messages
    conversationHistory.forEach(message => addMessage(message.role, message.content));
    updateButtonStates();
}

function saveConversationData() {
    localStorage.setItem('chatData', JSON.stringify({
        conversationId: currentConversationId,
        messages: conversationHistory
    }));
    updateButtonStates();
}

function addMessage(role, content) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', role);
    messageElement.textContent = `${role}: ${content}`;
    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function setButtonsDisabled(disabled) {
    conversationIdInput.disabled = disabled;
    sendButton.disabled = disabled;
    clearButton.disabled = disabled;
    saveButton.disabled = disabled;
    backButton.disabled = disabled;
    retryButton.disabled = disabled;
    userInput.disabled = disabled;
}

function showLoadingIndicator() {
    loadingIndicator.style.display = 'flex';
}

function hideLoadingIndicator() {
    loadingIndicator.style.display = 'none';
}

async function sendMessage(retryLast = false) {
    let userMessage;
    if (retryLast) {
        userMessage = conversationHistory[conversationHistory.length - 2].content;
    } else {
        userMessage = userInput.value.trim();
        if (!userMessage) return;
        addMessage('user', userMessage);
        conversationHistory.push({ role: 'user', content: userMessage });
    }

    saveConversationData();
    userInput.value = '';

    setButtonsDisabled(true);
    showLoadingIndicator();

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_KEY}`
            },
            body: JSON.stringify({
                model: 'aim-model',
                messages: conversationHistory,
                stream: true
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let assistantResponse = '';
        let assistantMessageElement = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data:')) {
                    const data = line.slice(5).trim();
                    if (data === '[DONE]') {
                        break;
                    }
                    try {
                        const parsed = JSON.parse(data);
                        if (parsed.choices && parsed.choices[0].delta && parsed.choices[0].delta.content) {
                            const content = parsed.choices[0].delta.content;
                            assistantResponse += content;
                            if (!assistantMessageElement) {
                                assistantMessageElement = document.createElement('div');
                                assistantMessageElement.classList.add('message', 'assistant');
                                chatContainer.appendChild(assistantMessageElement);
                            }
                            assistantMessageElement.textContent = `assistant: ${assistantResponse}`;
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }
                    } catch (error) {
                        console.error('Error parsing JSON:', error);
                    }
                }
            }
        }

        conversationHistory.push({ role: 'assistant', content: assistantResponse });
        saveConversationData();

    } catch (error) {
        console.error('Error:', error);
        addMessage('system', 'An error occurred while fetching the response.');
    } finally {
        setButtonsDisabled(false);
        hideLoadingIndicator();
        updateButtonStates();
    }
}

function clearChat() {
    chatContainer.innerHTML = '';
    conversationHistory = [];
    currentConversationId = generateConversationId();
    conversationIdInput.value = currentConversationId;
    saveConversationData();
}

async function saveConversation() {
    setButtonsDisabled(true);
    showLoadingIndicator();

    try {
        const response = await fetch(SAVE_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_KEY}`
            },
            body: JSON.stringify({
                conversation_id: currentConversationId,
                messages: conversationHistory
            })
        });

        if (!response.ok) {
            throw new Error('Failed to save conversation');
        }

        const result = await response.json();
        console.log('Conversation saved:', result);
        alert('Conversation saved successfully!');
    } catch (error) {
        console.error('Error saving conversation:', error);
        alert('Failed to save conversation. Please try again.');
    } finally {
        setButtonsDisabled(false);
        hideLoadingIndicator();
    }
}

function goBack() {
    if (conversationHistory.length >= 2) {
        conversationHistory.pop(); // Remove last assistant message
        conversationHistory.pop(); // Remove last user message
        chatContainer.innerHTML = ''; // Clear chat container
        conversationHistory.forEach(message => addMessage(message.role, message.content));
        saveConversationData();
    }
}

function retry() {
    if (conversationHistory.length >= 1) {
        conversationHistory.pop(); // Remove last assistant message
        chatContainer.lastElementChild.remove(); // Remove last message from chat container
        sendMessage(true); // Retry with the last user message
    }
}

function updateButtonStates() {
    const hasMessages = conversationHistory.length > 0;
    backButton.disabled = !hasMessages;
    retryButton.disabled = !hasMessages;
}

// Event Listeners
sendButton.addEventListener('click', () => sendMessage(false));
clearButton.addEventListener('click', clearChat);
saveButton.addEventListener('click', saveConversation);
backButton.addEventListener('click', goBack);
retryButton.addEventListener('click', retry);

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage(false);
    }
});

conversationIdInput.addEventListener('change', () => {
    currentConversationId = conversationIdInput.value;
    saveConversationData();
});

// Initialize the conversation when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initializeConversation();
    loadConversationHistory();
});