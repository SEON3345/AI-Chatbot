from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ = {'안녕하세요': '안녕하세요! 무엇을 도와드릴까요?', '파이썬 설치 방법': 'python.org에서 Python을 설치하세요.', '깃허브 사용 방법': '저장소를 만들고 파일을 커밋하세요.', '인공지능이란 무엇인가요': '데이터를 이용해 학습하고 추론하는 기술입니다.'}

class Chatbot:
    def __init__(self):
        self.questions = list(FAQ)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
        self.matrix = self.vectorizer.fit_transform(self.questions)

    def reply(self, message):
        scores = cosine_similarity(self.vectorizer.transform([message]), self.matrix)[0]
        idx = scores.argmax()
        return FAQ[self.questions[idx]] if scores[idx] >= 0.25 else '관련 답변을 찾지 못했습니다.'

if __name__ == '__main__':
    bot = Chatbot()
    while True:
        try:
            message = input('You: ')
        except (EOFError, KeyboardInterrupt):
            break
        if message.strip().lower() in ('exit', 'quit'): break
        print('Bot:', bot.reply(message))
