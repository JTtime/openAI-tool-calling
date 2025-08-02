import React, { useState } from 'react';
import { marked } from 'marked';
import '../App.css';

function Home() {
  const [inputText, setInputText] = useState('');
  const [responseText, setResponseText] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async () => {
    setIsLoading(true);
    setResponseText('');

    try {
      const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputText,
          conversation_history: [],
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let fullText = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n\n'); // SSE format

        for (let line of lines) {
          line = line.trim();
          if (line.startsWith('data:')) {
            const jsonStr = line.replace(/^data:\s*/, '');
            if (jsonStr) {
              try {
                const parsed = JSON.parse(jsonStr);
                if (parsed.content) {
                  fullText += parsed.content;
                  setResponseText(prev => prev + parsed.content); // live updating
                } else if (parsed.done) {
                  console.log('Streaming complete.');
                }
              } catch (e) {
                console.error('JSON parse error:', e);
              }
            }
          }
        }
      }
    } catch (err) {
      console.error('Error fetching stream:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Ask About a Product</h1>

      <div style={{ marginBottom: '1rem' }}>
        <input
          type="text"
          placeholder="Ask e.g. 'cheapest laptop'"
          value={inputText}
          onChange={e => setInputText(e.target.value)}
          style={{ padding: '0.5rem', width: '300px' }}
        />
        <button onClick={handleSend} disabled={isLoading} style={{ marginLeft: '1rem' }}>
          {isLoading ? 'Loading...' : 'Ask'}
        </button>
      </div>

      <div className="response-box" style={{ whiteSpace: 'pre-wrap', padding: '1rem', border: '1px solid #ccc' }}>
      <div dangerouslySetInnerHTML={{__html: marked(responseText || "")}}/>
  

      </div>
    </div>
  );
}

export default Home;
