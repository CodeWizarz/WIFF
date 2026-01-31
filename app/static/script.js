document.addEventListener('DOMContentLoaded', () => {
    const ingestBtn = document.getElementById('ingest-btn');
    const ingestContent = document.getElementById('ingest-content');
    const ingestStatus = document.getElementById('ingest-status');

    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatHistory = document.getElementById('chat-history');
    const contextContainer = document.getElementById('context-container');

    // --- Ingestion Logic ---
    ingestBtn.addEventListener('click', async () => {
        const text = ingestContent.value.trim();
        if (!text) return;

        setLoading(ingestBtn, true);
        ingestStatus.textContent = '';
        ingestStatus.className = 'status-msg';

        try {
            const response = await fetch('/api/v1/ingest/document', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    content: text,
                    source_type: 'document',
                    doc_metadata: { source: 'web_ui' }
                })
            });

            if (!response.ok) throw new Error('Ingestion failed');

            const data = await response.json();
            ingestStatus.textContent = `✓ Saved ${data.num_chunks} memory chunks.`;
            ingestStatus.classList.add('success');
            ingestContent.value = ''; // Clear input
        } catch (err) {
            console.error(err);
            ingestStatus.textContent = '❌ Error ingesting document.';
            ingestStatus.classList.add('error');
        } finally {
            setLoading(ingestBtn, false);
        }
    });

    // --- Chat Logic ---
    async function sendMessage() {
        const query = userInput.value.trim();
        if (!query) return;

        // 1. Add User Message
        appendMessage('user', query);
        userInput.value = '';

        // 2. Add "Thinking" Placeholder
        const thinkingId = appendMessage('system', 'Thinking taking a look at my memory...', true);

        try {
            const response = await fetch('/api/v1/agent/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            });

            const data = await response.json();

            // 3. Update Chat with Answer
            updateMessage(thinkingId, data.response);

            // 4. Update Context Panel
            renderContext(data.context_used);

        } catch (err) {
            console.error(err);
            updateMessage(thinkingId, "I'm having trouble connecting to my brain right now.");
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // --- Helpers ---
    function appendMessage(role, text, isThinking = false) {
        const id = Date.now().toString();
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        msgDiv.id = id;

        msgDiv.innerHTML = `
            <div class="avatar">${role === 'user' ? 'U' : 'AI'}</div>
            <div class="content">${text}</div>
        `;

        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return id;
    }

    function updateMessage(id, newText) {
        const msgDiv = document.getElementById(id);
        if (msgDiv) {
            msgDiv.querySelector('.content').textContent = newText;
        }
    }

    function renderDecisionResult(data) {
        const contentDiv = document.getElementById('decision-output');
        const resultContent = document.getElementById('decision-content');
        const auditLog = document.getElementById('audit-log');

        contentDiv.classList.remove('hidden');
        resultContent.innerHTML = '';
        auditLog.innerHTML = '';

        // 1. Meta Analysis (Top Level)
        const metaDiv = document.createElement('div');
        metaDiv.className = 'meta-analysis';

        let statusHtml = '';
        if (data.status) {
            const statusClass = data.status === 'approved' ? 'status-approved' :
                (data.status === 'rejected' ? 'status-rejected' : 'status-pending');
            statusHtml = `<div class="status-banner ${statusClass}">Status: ${data.status.toUpperCase()}</div>`;
        }

        metaDiv.innerHTML = `${statusHtml}
                             <strong>Recommendation:</strong> ${msgToHtml(data.meta_analysis)}`;
        resultContent.appendChild(metaDiv);

        // 2. Proposals
        data.proposals.forEach(p => {
            const pDiv = document.createElement('div');
            pDiv.className = `proposal-card ${p.id === data.selected_proposal_id ? 'selected' : ''}`;

            // Confidence Score
            const overallScore = p.scores.find(s => s.dimension === 'overall');
            const scoreVal = overallScore ? Math.round(overallScore.confidence * 100) : 0;

            let html = `
                <div class="proposal-header">
                    <h4>${p.title}</h4>
                    <span class="score-badge ${scoreVal > 80 ? 'high' : 'med'}">${scoreVal}% Confidence</span>
                </div>
                <p><strong>Rationale:</strong> ${p.rationale}</p>
                <div class="proposal-meta">
                    <span class="impact-badge ${p.impact}">${p.impact.toUpperCase()} IMPACT</span>
                </div>
            `;

            // Critique
            if (p.critique) {
                html += `<div class="critique-box">⚠️ <strong>Critic Note:</strong> ${p.critique}</div>`;
            }

            // Evidence
            if (p.supporting_evidence.length > 0) {
                html += `<div class="evidence-list"><small>Evidence:</small><ul>`;
                p.supporting_evidence.forEach(e => {
                    html += `<li>In "<em>${e.content_snippet.substring(0, 50)}...</em>" (Relevance: ${(e.relevance_score * 100).toFixed(0)}%)</li>`;
                });
                html += `</ul></div>`;
            }

            pDiv.innerHTML = html;
            resultContent.appendChild(pDiv);
        });

        // 3. Audit Trail
        if (data.audit_trail) {
            data.audit_trail.forEach(entry => {
                const entryDiv = document.createElement('div');
                entryDiv.className = 'audit-entry';

                const time = new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

                entryDiv.innerHTML = `
                    <div class="audit-time">${time}</div>
                    <div class="audit-agent">${entry.agent}</div>
                    <div class="audit-details"><strong>${entry.action}:</strong> ${entry.details}</div>
                `;
                auditLog.appendChild(entryDiv);
            });
        }
    }

    function msgToHtml(text) {
        return text.replace(/\n/g, '<br>');
    }

    function renderContext(chunks) {
        contextContainer.innerHTML = '';
        if (!chunks || chunks.length === 0) {
            contextContainer.innerHTML = '<div class="empty-state">No relevant memories found.</div>';
            return;
        }

        chunks.forEach(chunk => {
            const div = document.createElement('div');
            div.className = 'context-item';
            const scorePercent = Math.round((chunk.score || 0) * 100);
            div.innerHTML = `
                <div class="header">
                    <span>Relevance: ${scorePercent}%</span>
                    <span>ID: ${chunk.chunk_id}</span>
                </div>
                <div class="text">${chunk.content}</div>
            `;
            contextContainer.appendChild(div);
        });
    }

    function setLoading(btn, isLoading) {
        const spinner = btn.querySelector('.spinner');
        if (isLoading) {
            btn.disabled = true;
            spinner.classList.remove('hidden');
        } else {
            btn.disabled = false;
            spinner.classList.add('hidden');
        }
    }
});
