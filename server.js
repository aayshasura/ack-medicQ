// Load environment variables from .env file
require('dotenv').config();

const express = require('express');
const axios = require('axios');
const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');
const FormData = require('form-data');
const fs = require('fs').promises;
const path = require('path');
const { EventEmitter } = require('events');

const app = express();
const PORT = process.env.PORT || 3000;

// Create WebSocket server for real-time audio streaming
const wss = new WebSocket.Server({ port: PORT + 1 });

// Middleware
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
  } else {
    next();
  }
});

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));
app.use(express.static('public'));

// Configuration
const config = {
  zoom: {
    accountId: process.env.ZOOM_ACCOUNT_ID,
    clientId: process.env.ZOOM_CLIENT_ID,
    clientSecret: process.env.ZOOM_CLIENT_SECRET,
    webhookSecret: process.env.ZOOM_WEBHOOK_SECRET
  },
  n8n: {
    webhookUrl: process.env.N8N_WEBHOOK_URL || 'http://localhost:5678/webhook/transcript',
    apiKey: process.env.N8N_API_KEY
  },
  transcription: {
    service: process.env.TRANSCRIPTION_SERVICE || 'openai', // or 'assemblyai'
    chunkDuration: parseInt(process.env.AUDIO_CHUNK_DURATION) || 10000,
    openai: {
      apiKey: process.env.OPENAI_API_KEY,
      model: 'whisper-1'
    },
    assemblyai: {
      apiKey: process.env.ASSEMBLYAI_API_KEY,
      baseUrl: 'https://api.assemblyai.com/v2'
    }
  }
};

// In-memory storage
let zoomAccessToken = null;
let tokenExpiry = null;
const activeSessions = new Map();
const audioBuffers = new Map(); // Store audio chunks per session

class AudioTranscriptionSession extends EventEmitter {
  constructor(sessionId, meetingInfo, options = {}) {
    super();
    this.sessionId = sessionId;
    this.meetingInfo = meetingInfo;
    this.options = options;
    this.audioChunks = [];
    this.isProcessing = false;
    this.chunkCounter = 0;
    this.startTime = Date.now();
    
    console.log(`🎤 Created transcription session: ${sessionId}`);
  }

  async addAudioChunk(audioData) {
    try {
      this.chunkCounter++;
      const chunkId = `${this.sessionId}-chunk-${this.chunkCounter}`;
      
      console.log(`🎵 Received audio chunk ${this.chunkCounter} for session ${this.sessionId}`);
      console.log(`📊 Chunk size: ${audioData.length} bytes`);
      
      this.audioChunks.push({
        id: chunkId,
        data: audioData,
        timestamp: Date.now()
      });

      // Process chunks when we have enough data (every few seconds)
      if (this.audioChunks.length > 0 && !this.isProcessing) {
        await this.processAudioChunks();
      }

    } catch (error) {
      console.error(`❌ Error adding audio chunk: ${error.message}`);
      this.emit('error', error);
    }
  }

  async processAudioChunks() {
    if (this.isProcessing || this.audioChunks.length === 0) {
      return;
    }

    this.isProcessing = true;
    
    try {
      // Combine all current chunks
      const chunksToProcess = [...this.audioChunks];
      this.audioChunks = []; // Clear the buffer

      if (chunksToProcess.length === 0) {
        this.isProcessing = false;
        return;
      }

      console.log(`🔄 Processing ${chunksToProcess.length} audio chunks for transcription...`);

      // Combine audio chunks into single buffer
      const totalLength = chunksToProcess.reduce((sum, chunk) => sum + chunk.data.length, 0);
      const combinedAudio = Buffer.alloc(totalLength);
      let offset = 0;

      for (const chunk of chunksToProcess) {
        chunk.data.copy(combinedAudio, offset);
        offset += chunk.data.length;
      }

      console.log(`📊 Combined audio size: ${combinedAudio.length} bytes`);

      // Skip if audio is too small (likely silence or noise)
      if (combinedAudio.length < 1024) {
        console.log('⏭️  Skipping tiny audio chunk');
        this.isProcessing = false;
        return;
      }

      // Transcribe with OpenAI Whisper
      const transcript = await this.transcribeAudio(combinedAudio);
      
      if (transcript && transcript.text.trim()) {
        console.log(`📝 Transcription result: "${transcript.text}"`);
        
        // Send to N8N webhook
        await this.sendTranscriptToN8N(transcript);
        
        // Emit event for real-time updates
        this.emit('transcript', {
          sessionId: this.sessionId,
          transcript: transcript,
          chunkCount: chunksToProcess.length,
          timestamp: new Date().toISOString()
        });
      } else {
        console.log('🔇 No meaningful transcription result (likely silence)');
      }

    } catch (error) {
      console.error(`❌ Error processing audio chunks: ${error.message}`);
      this.emit('error', error);
    } finally {
      this.isProcessing = false;
    }
  }

  async transcribeAudio(audioBuffer) {
    try {
      if (config.transcription.service === 'assemblyai') {
        return await this.transcribeWithAssemblyAI(audioBuffer);
      } else {
        return await this.transcribeWithOpenAI(audioBuffer);
      }
    } catch (error) {
      console.error(`❌ ${config.transcription.service} transcription error:`, error.message);
      throw error;
    }
  }

  async transcribeWithOpenAI(audioBuffer) {
    console.log('🎤 Starting OpenAI Whisper transcription...');
    
    if (!config.transcription.openai.apiKey) {
      throw new Error('OpenAI API key not configured');
    }

    try {
      const formData = new FormData();
      formData.append('file', audioBuffer, {
        filename: 'audio.webm',
        contentType: 'audio/webm'
      });
      formData.append('model', config.transcription.openai.model);
      formData.append('language', 'en'); // Optional: specify language

      const response = await axios.post(
        'https://api.openai.com/v1/audio/transcriptions',
        formData,
        {
          headers: {
            'Authorization': `Bearer ${config.transcription.openai.apiKey}`,
            ...formData.getHeaders()
          }
        }
      );

      return {
        text: response.data.text,
        language: 'en', // OpenAI doesn't return language in response
        confidence: 1.0, // OpenAI doesn't provide confidence score
        service: 'openai-whisper',
        timestamp: Date.now()
      };
    } catch (error) {
      console.error('❌ OpenAI transcription error:', error.response?.data || error.message);
      throw error;
    }
  }

  async transcribeWithAssemblyAI(audioBuffer) {
    console.log('🎤 Starting AssemblyAI transcription...');
    
    if (!config.transcription.assemblyai.apiKey) {
      throw new Error('AssemblyAI API key not configured');
    }

    try {
      // Upload audio
      const uploadResponse = await axios.post(
        `${config.transcription.assemblyai.baseUrl}/upload`,
        audioBuffer,
        {
          headers: {
            'Authorization': config.transcription.assemblyai.apiKey,
            'Content-Type': 'application/octet-stream'
          }
        }
      );

      const audioUrl = uploadResponse.data.upload_url;

      // Start transcription
      const transcriptionResponse = await axios.post(
        `${config.transcription.assemblyai.baseUrl}/transcript`,
        {
          audio_url: audioUrl,
          language_code: 'en' // or detect automatically
        },
        {
          headers: {
            'Authorization': config.transcription.assemblyai.apiKey,
            'Content-Type': 'application/json'
          }
        }
      );

      const transcriptId = transcriptionResponse.data.id;
      let status = 'processing';

      // Poll for results
      while (status !== 'completed' && status !== 'error') {
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const resultResponse = await axios.get(
          `${config.transcription.assemblyai.baseUrl}/transcript/${transcriptId}`,
          {
            headers: {
              'Authorization': config.transcription.assemblyai.apiKey
            }
          }
        );

        status = resultResponse.data.status;
        
        if (status === 'completed') {
          return {
            text: resultResponse.data.text,
            language: resultResponse.data.language_code,
            confidence: resultResponse.data.confidence,
            service: 'assemblyai',
            timestamp: Date.now()
          };
        }
        
        if (status === 'error') {
          throw new Error('AssemblyAI transcription failed: ' + resultResponse.data.error);
        }
      }

      throw new Error('AssemblyAI transcription timed out');
    } catch (error) {
      console.error('❌ AssemblyAI transcription error:', error.response?.data || error.message);
      throw error;
    }
  }

  async sendTranscriptToN8N(transcript) {
    try {
      const payload = {
        sessionId: this.sessionId,
        timestamp: new Date().toISOString(),
        meetingInfo: this.meetingInfo,
        transcript: {
          text: transcript.text,
          language: transcript.language,
          confidence: transcript.confidence,
          service: transcript.service,
          chunkNumber: this.chunkCounter
        },
        session: {
          startTime: this.startTime,
          totalChunks: this.chunkCounter,
          duration: Date.now() - this.startTime
        }
      };

      const headers = {
        'Content-Type': 'application/json'
      };

      if (config.n8n.apiKey) {
        headers['Authorization'] = `Bearer ${config.n8n.apiKey}`;
      }

      console.log('📤 Sending transcript to N8N webhook...');
      const response = await axios.post(config.n8n.webhookUrl, payload, { 
        headers,
        timeout: 10000 // 10 second timeout
      });
      
      console.log('✅ Successfully sent to N8N:', response.status);
      return response.data;
      
    } catch (error) {
      console.error('❌ Error sending to N8N:', error.message);
      // Don't throw error - we don't want to stop transcription if N8N is down
    }
  }

  stop() {
    console.log(`⏹️  Stopping transcription session: ${this.sessionId}`);
    this.removeAllListeners();
  }
}

// Ensure temp directory exists
async function ensureTempDir() {
  const tempDir = path.join(__dirname, 'temp');
  try {
    await fs.mkdir(tempDir, { recursive: true });
  } catch (error) {
    console.log('Temp directory setup completed');
  }
}

// WebSocket connection handling for real-time audio
wss.on('connection', (ws, req) => {
  console.log('🔌 New WebSocket connection established');
  
  let currentSession = null;

  ws.on('message', async (message) => {
    try {
      const data = JSON.parse(message.toString());
      
      switch (data.type) {
        case 'start-session':
          console.log(`🚀 Starting new transcription session for meeting: ${data.meetingId}`);
          
          const sessionId = uuidv4();
          currentSession = new AudioTranscriptionSession(sessionId, {
            meetingId: data.meetingId,
            topic: data.topic || 'Real-time Meeting',
            participantId: data.participantId
          });

          activeSessions.set(sessionId, currentSession);

          // Listen for transcription events
          currentSession.on('transcript', (result) => {
            ws.send(JSON.stringify({
              type: 'transcript',
              data: result
            }));
          });

          currentSession.on('suggestions', (suggestions) => {
            ws.send(JSON.stringify({
              type: 'suggestions',
              data: suggestions
            }));
          });

          currentSession.on('error', (error) => {
            ws.send(JSON.stringify({
              type: 'error',
              error: error.message
            }));
          });

          ws.send(JSON.stringify({
            type: 'session-started',
            sessionId: sessionId,
            message: 'Transcription session started successfully'
          }));
          
          break;

        case 'audio-chunk':
          if (!currentSession) {
            ws.send(JSON.stringify({
              type: 'error',
              error: 'No active session. Please start a session first.'
            }));
            return;
          }

          // Convert base64 audio data to buffer
          const audioBuffer = Buffer.from(data.audioData, 'base64');
          await currentSession.addAudioChunk(audioBuffer);
          
          break;

        case 'stop-session':
          if (currentSession) {
            currentSession.stop();
            activeSessions.delete(currentSession.sessionId);
            currentSession = null;
            
            ws.send(JSON.stringify({
              type: 'session-stopped',
              message: 'Transcription session stopped'
            }));
          }
          break;

        default:
          console.log(`❓ Unknown message type: ${data.type}`);
      }

    } catch (error) {
      console.error('❌ WebSocket message error:', error.message);
      ws.send(JSON.stringify({
        type: 'error',
        error: 'Invalid message format'
      }));
    }
  });

  ws.on('close', () => {
    console.log('🔌 WebSocket connection closed');
    if (currentSession) {
      currentSession.stop();
      activeSessions.delete(currentSession.sessionId);
    }
  });

  ws.on('error', (error) => {
    console.error('❌ WebSocket error:', error.message);
  });
});

// Routes

// Serve client page for testing real-time transcription
app.get('/client', (req, res) => {
  res.send(`
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Zoom Transcription Client</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .status.connected { background-color: #d4edda; color: #155724; }
        .status.disconnected { background-color: #f8d7da; color: #721c24; }
        .transcript { background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .suggestions { background-color: #e6f7ff; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #1890ff; }
        button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
        .start { background-color: #28a745; color: white; }
        .stop { background-color: #dc3545; color: white; }
        input, select { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }
        .audio-controls { background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>🎤 Real-time Zoom Transcription</h1>
    
    <div id="status" class="status disconnected">Disconnected</div>
    
    <div class="audio-controls">
        <h3>Meeting Information</h3>
        <input type="text" id="meetingId" placeholder="Zoom Meeting ID" value="123456789">
        <input type="text" id="meetingTopic" placeholder="Meeting Topic" value="Test Meeting">
        <br><br>
        
        <h3>Audio Controls</h3>
        <button id="startBtn" class="start">Start Recording & Transcription</button>
        <button id="stopBtn" class="stop" disabled>Stop Recording</button>
        <br><br>
        
        <div>
            <strong>Status:</strong> <span id="recordingStatus">Ready</span><br>
            <strong>Session:</strong> <span id="sessionId">None</span>
        </div>
    </div>
    
    <div id="transcripts">
        <h3>📝 Live Transcriptions</h3>
        <div id="transcript-list"></div>
    </div>

    <script>
        let ws = null;
        let mediaRecorder = null;
        let audioStream = null;
        let recordingInterval = null;
        
        const statusDiv = document.getElementById('status');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const recordingStatus = document.getElementById('recordingStatus');
        const sessionIdSpan = document.getElementById('sessionId');
        const transcriptList = document.getElementById('transcript-list');
        
        function connectWebSocket() {
            ws = new WebSocket('ws://' + window.location.hostname + ':${PORT + 1}');
            
            ws.onopen = () => {
                statusDiv.textContent = 'Connected to transcription server';
                statusDiv.className = 'status connected';
                console.log('✅ Connected to WebSocket server');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log('📨 Received:', data);
                
                switch(data.type) {
                    case 'session-started':
                        sessionIdSpan.textContent = data.sessionId;
                        recordingStatus.textContent = 'Session Active';
                        break;
                        
                    case 'transcript':
                        addTranscript(data.data);
                        break;
                        
                    case 'suggestions':
                        displaySuggestions(data.data);
                        break;
                        
                    case 'error':
                        console.error('❌ Server error:', data.error);
                        alert('Error: ' + data.error);
                        break;
                        
                    case 'session-stopped':
                        recordingStatus.textContent = 'Session Stopped';
                        sessionIdSpan.textContent = 'None';
                        break;
                }
            };
            
            ws.onclose = () => {
                statusDiv.textContent = 'Disconnected from server';
                statusDiv.className = 'status disconnected';
                console.log('❌ WebSocket connection closed');
            };
            
            ws.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
            };
        }
        
        function addTranscript(transcriptData) {
            const div = document.createElement('div');
            div.className = 'transcript';
            div.innerHTML = \`
                <div><strong>[\${new Date(transcriptData.timestamp).toLocaleTimeString()}]</strong></div>
                <div>\${transcriptData.transcript.text}</div>
                <small>Language: \${transcriptData.transcript.language} | Confidence: \${(transcriptData.transcript.confidence * 100).toFixed(1)}%</small>
            \`;
            transcriptList.insertBefore(div, transcriptList.firstChild);
        }
        
        function displaySuggestions(suggestions) {
            const div = document.createElement('div');
            div.className = 'suggestions';
            div.innerHTML = \`
                <strong>💡 Suggested Questions:</strong>
                <ul>\${suggestions.map(q => '<li>' + q + '</li>').join('')}</ul>
            \`;
            transcriptList.insertBefore(div, transcriptList.firstChild);
        }
        
        async function startRecording() {
            try {
                recordingStatus.textContent = 'Requesting microphone access...';
                
                audioStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true
                    } 
                });
                
                mediaRecorder = new MediaRecorder(audioStream, {
                    mimeType: 'audio/webm;codecs=opus'
                });
                
                const audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                
                // Send audio chunks every 5 seconds
                mediaRecorder.onstop = () => {
                    if (audioChunks.length > 0) {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        const reader = new FileReader();
                        reader.onload = () => {
                            const base64Audio = reader.result.split(',')[1];
                            if (ws && ws.readyState === WebSocket.OPEN) {
                                ws.send(JSON.stringify({
                                    type: 'audio-chunk',
                                    audioData: base64Audio
                                }));
                            }
                        };
                        reader.readAsDataURL(audioBlob);
                    }
                };
                
                // Start session
                ws.send(JSON.stringify({
                    type: 'start-session',
                    meetingId: document.getElementById('meetingId').value,
                    topic: document.getElementById('meetingTopic').value,
                    participantId: 'web-client'
                }));
                
                // Record in 5-second chunks
                mediaRecorder.start();
                recordingInterval = setInterval(() => {
                    if (mediaRecorder.state === 'recording') {
                        mediaRecorder.stop();
                        setTimeout(() => {
                            if (mediaRecorder && audioStream) {
                                mediaRecorder.start();
                            }
                        }, 100);
                    }
                }, 5000);
                
                startBtn.disabled = true;
                stopBtn.disabled = false;
                recordingStatus.textContent = 'Recording & Transcribing...';
                
            } catch (error) {
                console.error('❌ Error starting recording:', error);
                alert('Error accessing microphone: ' + error.message);
                recordingStatus.textContent = 'Error';
            }
        }
        
        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
            
            if (recordingInterval) {
                clearInterval(recordingInterval);
            }
            
            if (audioStream) {
                audioStream.getTracks().forEach(track => track.stop());
            }
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'stop-session'
                }));
            }
            
            startBtn.disabled = false;
            stopBtn.disabled = true;
            recordingStatus.textContent = 'Stopped';
        }
        
        startBtn.addEventListener('click', startRecording);
        stopBtn.addEventListener('click', stopRecording);
        
        // Connect on page load
        connectWebSocket();
    </script>
</body>
</html>
  `);
});

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    activeSessions: activeSessions.size,
    websocketPort: PORT + 1,
    configuration: {
      hasZoomCredentials: !!(config.zoom.accountId && config.zoom.clientId && config.zoom.clientSecret),
      hasOpenAIKey: !!config.transcription.openai.apiKey,
      transcriptionService: config.transcription.service,
      chunkDuration: config.transcription.chunkDuration
    }
  });
});

// Get active sessions
app.get('/sessions', (req, res) => {
  const sessions = Array.from(activeSessions.entries()).map(([id, session]) => ({
    sessionId: id,
    meetingInfo: session.meetingInfo,
    startTime: session.startTime,
    chunkCounter: session.chunkCounter,
    isProcessing: session.isProcessing
  }));
  
  res.json({
    totalSessions: sessions.length,
    sessions: sessions
  });
});

// Test endpoint for N8N webhook
app.post('/test-n8n', async (req, res) => {
  try {
    const testPayload = {
      sessionId: 'test-session',
      timestamp: new Date().toISOString(),
      meetingInfo: {
        meetingId: '123456789',
        topic: 'Test Meeting'
      },
      transcript: {
        text: 'This is a test transcript from the Zoom transcription service.',
        language: 'en',
        confidence: 0.95,
        service: 'openai-whisper'
      }
    };

    const response = await axios.post(config.n8n.webhookUrl, testPayload, {
      headers: {
        'Content-Type': 'application/json'
      },
      timeout: 10000
    });

    res.json({
      status: 'success',
      message: 'Test payload sent to N8N successfully',
      response: {
        status: response.status,
        data: response.data
      }
    });

  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: 'Failed to send test payload to N8N',
      error: error.message
    });
  }
});

// Test endpoint for AssemblyAI
app.post('/test-assemblyai', async (req, res) => {
  try {
    if (!config.transcription.assemblyai.apiKey) {
      return res.status(400).json({
        status: 'error',
        message: 'AssemblyAI API key not configured'
      });
    }

    // Test AssemblyAI connectivity
    const response = await axios.get(`${config.transcription.assemblyai.baseUrl}/transcript`, {
      headers: {
        'Authorization': config.transcription.assemblyai.apiKey
      }
    });

    res.json({
      status: 'success',
      message: 'AssemblyAI connection test successful',
      accountTier: response.data?.tier || 'unknown'
    });

  } catch (error) {
    console.error('❌ AssemblyAI test failed:', error.message);
    res.status(500).json({
      status: 'error',
      message: 'AssemblyAI test failed',
      error: error.message
    });
  }
});

// Receive suggested questions back from n8n
app.post('/suggested-questions', async (req, res) => {
  try {
    const { sessionId, suggestions } = req.body;

    if (!sessionId || !Array.isArray(suggestions)) {
      return res.status(400).json({ error: 'Invalid payload' });
    }

    // Forward suggestions to the active WebSocket client (if session exists)
    const session = activeSessions.get(sessionId);
    if (session) {
      session.emit('suggestions', suggestions);
    }

    console.log(`💡 Suggestions received for session ${sessionId}:`, suggestions);
    res.json({ status: 'ok' });

  } catch (err) {
    console.error('❌ Error receiving suggestions:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// Error handling
app.use((error, req, res, next) => {
  console.error('💥 Unhandled error:', error);
  res.status(500).json({ error: 'Internal server error', details: error.message });
});

// Start server
async function startServer() {
  try {
    await ensureTempDir();
    
    app.listen(PORT, () => {
      console.log(`\n🚀 Real-time Zoom Transcription Server`);
      console.log(`📊 HTTP Server: http://localhost:${PORT}`);
      console.log(`🔌 WebSocket Server: ws://localhost:${PORT + 1}`);
      console.log(`🧪 Test Client: http://localhost:${PORT}/client`);
      console.log(`📊 Health check: http://localhost:${PORT}/health`);
      console.log(`📝 Active sessions: http://localhost:${PORT}/sessions`);
      console.log(`🧪 Test N8N webhook: POST http://localhost:${PORT}/test-n8n`);
      
      console.log('\n🔄 Audio Processing Flow:');
      console.log('   Zoom Meeting → WebSocket → Audio Chunks → OpenAI Whisper → N8N Webhook');
      
      console.log('\n📝 Required Environment Variables:');
      console.log('   - OPENAI_API_KEY (required)');
      console.log('   - N8N_WEBHOOK_URL (required)');
      console.log('   - ZOOM_ACCOUNT_ID (optional, for meeting info)');
      console.log('   - ZOOM_CLIENT_ID (optional)');
      console.log('   - ZOOM_CLIENT_SECRET (optional)');
      
      const issues = [];
      if (!config.transcription.openai.apiKey) {
        issues.push('⚠️  OpenAI API key not configured');
      }
      if (!config.n8n.webhookUrl.includes('http')) {
        issues.push('⚠️  N8N webhook URL not properly configured');
      }
      
      if (issues.length > 0) {
        console.log('\n' + issues.join('\n'));
      } else {
        console.log('\n✅ Core transcription services configured properly');
      }
      
      console.log(`\n🎤 WebSocket server listening on port ${PORT + 1} for real-time audio`);
    });
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
}

startServer();