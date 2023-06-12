import speech_recognition as sr


class VoiceASR:
    def __init__(self):
        self.r = sr.Recognizer()

    def transcribe_from_file(self, audio_path):
        with sr.AudioFile(audio_path) as source:
            audio_data = self.r.record(source)
            text = self.r.recognize_google(audio_data, language="cs-CZ")
            return text

    def transcribe_from_samples(self, audio_samples):
        audio_data = sr.AudioData(audio_samples, sample_rate=16000, sample_width=2)
        text = self.r.recognize_google(audio_data, language="cs-CZ")
        return text