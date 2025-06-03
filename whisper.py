
from streamlit_mic_recorder import mic_recorder
import streamlit as st
import io
from openai import OpenAI
import dotenv
import os


def whisper_stt(openai_api_key=os.getenv('OPENAI_API_KEY'), start_prompt="🎙️", stop_prompt="🔴", just_once=True,
               use_container_width=False, language=None, callback=None, args=(), kwargs=None, key=None):
    if not 'openai_client' in st.session_state:
        dotenv.load_dotenv()
        st.session_state.openai_client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
    if not '_last_speech_to_text_transcript_id' in st.session_state:
        st.session_state._last_speech_to_text_transcript_id = 0
    if not '_last_speech_to_text_transcript' in st.session_state:
        st.session_state._last_speech_to_text_transcript = None
    if key and not key + '_output' in st.session_state:
        st.session_state[key + '_output'] = None
    # Add this CSS injection just before calling mic_recorder
    st.markdown("""
    <style>
    .mic-container {
        display: flex;
        justify-content: flex-end; /* align right */
        margin-bottom: 0.1rem;
    }
    .mic-container button {
        font-size: 1.5rem !important;  /* bigger font */
        padding: 1rem 2rem !important; /* bigger padding */
        transform: scale(1.3);         /* scale button */
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

    # Wrap your mic_recorder call inside this div
    st.markdown('<div class="mic-container">', unsafe_allow_html=True)
    audio = mic_recorder(start_prompt=start_prompt, stop_prompt=stop_prompt, just_once=just_once,
                        use_container_width=use_container_width, key=key)
    st.markdown('</div>', unsafe_allow_html=True)

    new_output = False
    if audio is None:
        output = None
    else:
        id = audio['id']
        new_output = (id > st.session_state._last_speech_to_text_transcript_id)
        if new_output:
            output = None
            st.session_state._last_speech_to_text_transcript_id = id
            audio_bio = io.BytesIO(audio['bytes'])
            audio_bio.name = 'audio.mp3'
            success = False
            err = 0
            while not success and err < 3:  # Retry up to 3 times in case of OpenAI server error.
                try:
                    transcript = st.session_state.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_bio,
                        language=language
                    )
                except Exception as e:
                    print(str(e))  # log the exception in the terminal
                    err += 1
                else:
                    success = True
                    output = transcript.text
                    st.session_state._last_speech_to_text_transcript = output
        elif not just_once:
            output = st.session_state._last_speech_to_text_transcript
        else:
            output = None

    if key:
        st.session_state[key + '_output'] = output
    if new_output and callback:
        callback(*args, **(kwargs or {}))
    return output