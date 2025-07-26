"""UI configuration, theme settings, and constants for BoardRAG application."""

import gradio as gr

# JavaScript for localStorage management
STORAGE_SET_JS = '''
function(session_id) {
    try {
        console.log('[DEBUG] STORAGE_SET_JS called with:', session_id);
        localStorage.setItem('boardrag_session_id', session_id);
        console.log('[DEBUG] Set session ID:', session_id);
    } catch (error) {
        console.error('[DEBUG] Error setting session ID:', error);
    }
}
'''

# JavaScript for setting access state in localStorage
STORAGE_SET_ACCESS_JS = '''
function(access_state) {
    try {
        console.log('[DEBUG] STORAGE_SET_ACCESS_JS called with:', access_state);
        if (access_state === "user" || access_state === "admin") {
            localStorage.setItem('boardrag_access_state', access_state);
            console.log('[DEBUG] Set access state:', access_state);
        } else {
            console.log('[DEBUG] Skipping access state - invalid:', access_state);
        }
    } catch (error) {
        console.error('[DEBUG] Error setting access state:', error);
    }
}
'''

# JavaScript for reading existing sessions from localStorage
STORAGE_READ_JS = '''
function() {
    console.log('[DEBUG] STORAGE_READ_JS executed - attempting localStorage read');
    
    try {
        // Check if localStorage is available
        if (typeof(Storage) === "undefined") {
            console.error('[DEBUG] localStorage not supported by browser');
            const fallbackId = 'no-storage-' + Date.now();
            return [fallbackId, 'none'];
        }
        
        // Try to read existing session from localStorage
        let sessionId = localStorage.getItem('boardrag_session_id');
        const accessState = localStorage.getItem('boardrag_access_state');
        
        console.log('[DEBUG] Raw localStorage values:');
        console.log('[DEBUG] - boardrag_session_id:', sessionId);
        console.log('[DEBUG] - boardrag_access_state:', accessState);
        
        // If we have an existing session, use it
        if (sessionId && sessionId.length > 0 && sessionId !== 'null' && sessionId !== 'undefined') {
            console.log('[DEBUG] Found valid existing session ID:', sessionId);
            const finalAccessState = accessState || 'none';
            console.log('[DEBUG] Restoring session:', sessionId, 'with access:', finalAccessState);
            return [sessionId, finalAccessState];
        }
        
        // No existing session - generate new one
        console.log('[DEBUG] No valid session found, generating new one');
        sessionId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
        
        // Save the new session ID immediately
        localStorage.setItem('boardrag_session_id', sessionId);
        console.log('[DEBUG] Generated and saved new session ID:', sessionId);
        
        const finalAccessState = accessState || 'none';
        console.log('[DEBUG] Returning new session:', [sessionId, finalAccessState]);
        return [sessionId, finalAccessState];
        
    } catch (error) {
        console.error('[DEBUG] localStorage error:', error);
        // Generate fallback session ID
        const fallbackId = 'error-fallback-' + Date.now();
        console.log('[DEBUG] Using error fallback session ID:', fallbackId);
        return [fallbackId, 'none'];
    }
}
'''

# Introduction text
INTRO_STRING = """
# 🎲 BoardRAG
"""

# Custom CSS for the interface
THEME_CSS = """
.main-content {
    max-width: 1200px;
    margin: 0 auto;
}

.chat-column {
/*    min-height: 600px;*/
}

.custom-chatbot {
    font-size: 12px;  

}

/* Prompt list styling */
#prompt-radio input[type="radio"] {
    /* hide native circle */
    display: none;
}

#prompt-radio label {
    display: block;
    width: 100%;
    padding: 6px 10px;
    margin: 4px 0;
    border: 1px solid rgba(255,255,255,0.6);
    border-radius: 4px;
    background: transparent;
    font-size: 12px;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: #fff;
    box-sizing: border-box;
}

#prompt-radio label:hover,
#prompt-radio input[type="radio"]:checked + label {
    background: rgba(255,255,255,0.15);
}

/* Ensure selected radio stays highlighted */
#prompt-radio label:has(input:checked),
#prompt-radio input[type="radio"]:checked + label {
    background: rgba(255,255,255,0.25);
}

/* Bookmarks panel with height limit and scrolling */
#prompt-radio {
    max-height: 800px;
    overflow-y: auto;
    overflow-x: hidden;
}

/* Custom scrollbar styling for bookmarks panel */
#prompt-radio::-webkit-scrollbar {
    width: 6px;
}

#prompt-radio::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
}

#prompt-radio::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.3);
    border-radius: 3px;
}

#prompt-radio::-webkit-scrollbar-thumb:hover {
    background: rgba(255,255,255,0.5);
}

/* Custom scrollbar styling for chat box */
.custom-chatbot::-webkit-scrollbar {
    width: 6px;
}

.custom-chatbot::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
}

.custom-chatbot::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.3);
    border-radius: 3px;
}

.custom-chatbot::-webkit-scrollbar-thumb:hover {
    background: rgba(255,255,255,0.5);
}

/* Force bubble padding & gap overrides */
.custom-chatbot .message {
    padding: 0px 0px !important;
}

.custom-chatbot .message-content {
    margin: 0px 0px !important;
}

/* Gradio 4 puts text into .md; tighten that too */
.custom-chatbot .md {
    font-size: 14px !important;
    line-height: 3 !important;
    padding: 4 !important;
    margin: 0 !important;
}

.custom-chatbot .bubble {
    margin: 4px 4px !important;
}

.custom-chatbot .bot-row {
    width: 96% !important;
    max-width: 96% !important;
    margin: 4px 4px !important;
}


/* Ensure container gap override sticks even with Tailwind gap utilities */
.custom-chatbot > div {
    gap: 0px !important;
}

/* Remove left/right padding that Gradio adds around the scroll area */
.custom-chatbot > div:first-child {
    padding-left: 0 !important;
    padding-right: 0 !important;
}

/* Remove default horizontal margins on each bubble */
.custom-chatbot .message {
    margin-left: 0 !important;
    margin-right: 0 !important;
}

.input-row {
    margin-top: 10px;
}

.progress-indicator {
    margin: 10px 0;
}

/* Mobile responsive adjustments */
@media (max-width: 768px) {
    .main-content {
        padding: 2px;
    }
    
}

/* ------------------------------------------------------------------
   Disable Gradio "processing" flash on the bookmarks accordion only
   ------------------------------------------------------------------*/
#prompt-radio[aria-busy="true"],
.gr-accordion:has(#prompt-radio)[aria-busy="true"] {
    opacity: 1 !important; /* cancel dimming */
    animation: none !important; /* cancel pulse */
}

#prompt-radio .gr-skeleton,
#prompt-radio .animate-pulse {
    display: none !important; /* hide shimmer bar */
}

#prompt-radio > .absolute.inset-0 {
    display: none !important; /* remove click-blocking overlay */
} 

"""

def get_config_info():
    """Get configuration information for display."""
    import config
    return f"""
## ⚙️ Configuration

**Provider:** {config.LLM_PROVIDER}  
**Generator Model:** {config.GENERATOR_MODEL}  
**Embedder Model:** {config.EMBEDDER_MODEL}  
**Chunk Size:** {config.CHUNK_SIZE}  
**Chunk Overlap:** {config.CHUNK_OVERLAP}  
**Web Search:** {"Enabled" if config.ENABLE_WEB_SEARCH else "Disabled"}  
**Data Path:** {config.DATA_PATH}  
**ChromaDB Path:** {config.CHROMA_PATH}  

## 🔐 Access Control

**USER_PW:** {"✅ Set" if config.USER_PW else "❌ Missing"}  
**ADMIN_PW:** {"✅ Set" if config.ADMIN_PW else "❌ Missing"}  
"""


def create_theme():
    """Create and return the Gradio theme with smaller base text size."""
    return gr.themes.Default(
        font=[gr.themes.GoogleFont("Georgia"), "Arial", "sans-serif"],
        spacing_size=gr.themes.sizes.spacing_sm,
        radius_size=gr.themes.sizes.radius_none,
    ) 