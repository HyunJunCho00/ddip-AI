import pandas as pd
import numpy as np
import random

# ---------------------
# 1️⃣ 키워드 정의
# ---------------------
lose_keywords = [
    '두고', '놓고', '잃어', '분실', '찾아', '남겨', '흘렸', '떨어뜨렸', '어딘가에 놔두고', '분실한 것 같', '기억이 안 나',
    '안 가져왔', '어디 뒀는지 모르겠', '두고 온 것 같', '안 챙겼', '깜빡하고 놓고', '놓고 나온 것 같'
]

high_value_keywords = [
    '애플펜슬', '아이패드', '노트북', '폰', '지갑', '카드', '갤럭시탭', '에어팟', '워치', '카메라', '맥북',
    '갤럭시버즈', '아이폰', '태블릿', '갤럭시북', '블루투스 이어폰', '아이팟', '고프로', '아이디카드', '학생증 카드'
]

medium_value_items = [
    '충전기', '우산', '텀블러', '전공 책', '필통', '모자', '보조배터리', '마우스', '학생증', '목도리',
    '에코백', '안경', '안경집', 'USB', '책', '노트', '이어폰', '헤어밴드', '장갑', '볼펜', '연필 케이스', '명찰'
]

locations = [
    '중앙도서관 1층 열람실', '백호관(농대) 101호', 'IT3호관', '공대 12호관', '글로벌플라자 1층',
    '복지관 학생식당', '제1학생회관', '경상대학 건물', '사회과학대학 열람실', '백양로', '일청담 근처 벤치',
    '대강당', '첨성관(기숙사) 로비', '중도 신관','중도 구관','보람관','누리관','공대6호관','인문대','경상대','자연대','미융관',
    '미래융합관','조형관','사과대','법대','경상대','4합','간호대','테크노빌딩','정센','정보센터','GS25','세븐일레븐','공식당','교직원식당',
    '첨성 카페테리아','복지관 교직원 식당','공대 7호관','공대8호관','공대9호관','공대10호관','공대1호관','오도','IT1호관','IT2호관','IT5호관','융복합관','융복','IT4호관',
    '조은문(북문) 근처 카페'
]
colors = [
    '검은색', '흰색', '파란색', '실버', '스페이스 그레이', '남색',
    '하늘색', '분홍색', '연두색', '회색', '노란색', '빨간색', '갈색', '보라색', '청록색', '베이지', '크림색'
]

find_verbs = [
    '보신 분 계신가요?', '찾아주시면 사례하겠습니다', '아직 있을까요?', '습득하신 분 연락주세요', '보이면 알려주세요 제발ㅠㅠ',
    '혹시 주우신 분 있으신가요?', '보셨다면 연락 부탁드립니다', '혹시 맡아주신 분 계신가요?', '습득하신 분 계시면 꼭 연락 주세요!',
    '혹시 어디 보관되어 있는지 아시나요?', '지금도 거기 있을까요?', '꼭 찾아야 하는데 도와주세요ㅠㅠ', '잃어버린 것 같아요.. 도와주세요!'
]

weather_bonus = {'맑음': 0, '비': 500, '눈': 500, '추움': 300, '더움': 400, '태풍': 1000, '황사': 200}
weather_conditions = list(weather_bonus.keys())
weather_probabilities = [0.6, 0.1, 0.05, 0.1, 0.1, 0.02, 0.03]

# ---------------------
# 2️⃣ 난이도 분류
# ---------------------
def classify_difficulty(text):
    if any(word in text for word in lose_keywords):
        if any(word in text for word in high_value_keywords):
            return '상'
        else:
            return '중'
    return '하'

# ---------------------
# 3️⃣ 한글 오탈자 변형 (개선된 버전)
# ---------------------
import random

# 한글 자모 분해/결합을 위한 설정
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG_LIST = [''] + ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 키보드에서 인접한 자모들 (일부 예시)
CHOSUNG_NEIGHBORS = {'ㄱ': 'ㄲㅋ', 'ㄷ': 'ㄸㅌ', 'ㅂ': 'ㅃㅍ', 'ㅅ': 'ㅆ', 'ㅈ': 'ㅉㅊ', 'ㅇ': 'ㅎㅁ'}
JUNGSUNG_NEIGHBORS = {'ㅏ': 'ㅑㅣ', 'ㅓ': 'ㅕㅔ', 'ㅗ': 'ㅛㅜ', 'ㅜ': 'ㅠㅗ', 'ㅡ': 'ㅣㅜ', 'ㅣ': 'ㅏㅡ'}
JONGSUNG_NEIGHBORS = {'ㄱ': 'ㄲㄳ', 'ㄴ': 'ㄵㄶ', 'ㄹ': 'ㄺㄻㄼㄽㄾㄿㅀ'}

def decompose(syllable):
    """한글 음절을 초성, 중성, 종성으로 분해"""
    if '가' <= syllable <= '힣':
        code = ord(syllable) - ord('가')
        ch_idx = code // (21 * 28)
        jung_idx = (code % (21 * 28)) // 28
        jong_idx = code % 28
        return CHOSUNG_LIST[ch_idx], JUNGSUNG_LIST[jung_idx], JONGSUNG_LIST[jong_idx]
    return syllable, '', ''

def compose(chosung, jungsung, jongsung=''):
    """초성, 중성, 종성을 한글 음절로 결합"""
    try:
        ch_idx = CHOSUNG_LIST.index(chosung)
        jung_idx = JUNGSUNG_LIST.index(jungsung)
        jong_idx = JONGSUNG_LIST.index(jongsung)
        return chr(ord('가') + (ch_idx * 21 * 28) + (jung_idx * 28) + jong_idx)
    except:
        return chosung # 한글 자모가 아니면 그대로 반환

def typo_variants_korean(word):
    """현실적인 오타를 생성하는 함수"""
    if len(word) < 2:
        return [word]

    variants = {word}
    chars = list(word)

    # 1. 특정 글자의 자/모음 바꾸기 (가장 흔한 오타 유형)
    for _ in range(3): # 다양한 변형을 위해 3번 시도
        idx = random.randrange(len(chars))
        char = chars[idx]

        decomposed = decompose(char)
        if not decomposed[1]: continue # 한글이 아니면 스킵

        ch, jung, jong = decomposed

        # 1-1. 모음(중성) 바꾸기 (예: ㅔ -> ㅐ)
        if jung in JUNGSUNG_NEIGHBORS:
            new_jung = random.choice(JUNGSUNG_NEIGHBORS[jung])
            new_char = compose(ch, new_jung, jong)
            variants.add(word[:idx] + new_char + word[idx+1:])

        # 1-2. 자음(초성) 바꾸기 (예: ㄱ -> ㅋ)
        if ch in CHOSUNG_NEIGHBORS:
            new_ch = random.choice(CHOSUNG_NEIGHBORS[ch])
            new_char = compose(new_ch, jung, jong)
            variants.add(word[:idx] + new_char + word[idx+1:])

    # 2. 받침(종성) 빼기 (예: 애플 -> 애프)
    idx = random.randrange(len(chars))
    ch, jung, jong = decompose(chars[idx])
    if jong: # 받침이 있는 경우에만
        new_char = compose(ch, jung, '') # 받침 제거
        variants.add(word[:idx] + new_char + word[idx+1:])


    return list(variants)

# ---------------------
# 4️⃣ 문장 생성 함수
# ---------------------
def generate_sentence(item_list, difficulty):
    item = random.choice(item_list)
    location = random.choice(locations)
    title = f"{item} 분실했습니다ㅠㅠ"
    if random.random() < 0.5:
        title = f"{location}에서 {item} 보신 분?"
    desc_list = [f"{location}에 {item}을 놓고 온 것 같아요.."]
    if random.random() > 0.4:
        desc_list.append(f"{random.choice(colors)} 색상인데,")
    desc_list.append(random.choice(find_verbs))
    content = ' '.join(desc_list)
    return {'요청글 제목': title, '요청글 내용': content, '난이도': difficulty}

# ---------------------
# 하 난이도 요청글 자동 생성 함수
# ---------------------

objects = ['비둘기', '고양이', '택배', '우산', '의자']
phenomena = ['소리', '냄새', '비명소리', '비둘기', '물 웅덩이']
tasks = ['의자 조립', '이사 도움', '문서 프린트', '컴퓨터 설정']
borrow_items = ['공기계', '노트북', '충전기', '보조배터리']



# 🧩 1. 하 난이도용 패턴 정의 (분실 언급 금지)
low_value_patterns = [
    "{place} 지금 사람 많나요?",
    "{place} 예약되어 있나요?",
    "{place}에 {object} 있나요?",
    "{place}에 {phenomenon} 생김 → 사진 있으신 분?",
    "{place} 지금 뭐 행사하나요?",
    "{item} 하루만 빌려주실 분 계신가요?",
    "{task} 도와주실 분 찾습니다",
    "택배 반품 도와주실 분 계신가요?",
    "{place} 근처에 {item} 떨어진 거 못 보셨나요?",
    "{place}에 이상한 {phenomenon} 있었는데 혹시 보신 분?",
    "{place} 가보신 분 계신가요? 지금 뭐 있나요?",
    "{place} 주변 CCTV 있나요?",

    # ✅ 추가 패턴
    "{place} 에어컨 시원하게 나오나요?",
    "{place} 조용히 공부할 수 있나요?",
    "{place} 콘센트 있는 자리 많나요?",
    "{place}에 와이파이 잘 되나요?",
    "{place}에 쉴 자리 있나요?",
    "{place}에 프린트 가능하나요?",
    "{place} 주차 가능한가요?",
    "{place} 근처에 {object} 무리 본 사람 있나요?",
    "{place}에 {object} 자주 나타나나요?",
    "{place}에 불빛 번쩍이던데 무슨 일인가요?",
    "{place}에 소란스럽던데 무슨 일인지 아시는 분?",
    "{place} 근처 {phenomenon} 있었어요. 확인해보신 분?",
    "{place}에 비둘기 엄청 많던데 이유 아시는 분?",
    "{place} 근처 자판기 있나요?",
    "{place} 쾌적한가요?",
    "{place} 지금 운영 중인가요?",
    "{place}랑 {place2} 중 어디가 더 한산하나요?",
    "{place} 요즘에 북적이나요?",
    "{place}에 무슨 안내 방송 나왔던데 아시는 분?",
    "{place} 근처에서 이상한 {phenomenon} 들리던데 혹시 보신 분?",
    "{place} 분위기 어떤가요?",
    "{place} 식당 지금 줄 길어요?",
    "{place} 근처에 공부할 공간 있을까요?",
    "{place} 지금 오픈했는지 아시는 분?",
    "{place} 근처 앉을 수 있는 벤치 있나요?",
    "{place} 근처 쓰레기통 어디 있나요?",
    "{place} 근처 프린터 고장났던데, 다른 데 아시나요?",
    "{place} 근처 음료 자판기 뭐 있나요?",
    "{place} 앞에 긴 줄 있던데 무슨 줄인가요?",
    "{place}에서 사진 찍기 좋은 장소 아시나요?"
]

# 🧩 2. 요청 생성에 쓰일 변수들
objects = ['비둘기', '고양이', '택배', '우산', '의자', '강아지', '벌레', '드론', '비둘기 무리', '야옹이']
phenomena = ['소리', '냄새', '비명소리', '비둘기', '물 웅덩이', '연기', '진동', '화재 경보음', '울음소리']
tasks = ['의자 조립', '이사 도움', '문서 프린트', '컴퓨터 설정', '책상 조립', '노트북 셋업', '데이터 백업', 'PPT 만들기']
borrow_items = ['공기계', '노트북', '충전기', '보조배터리', '프린트용지', '노트', 'USB', '마우스', '포스트잇']

# 🧩 3. 요청글 생성 함수 (하 난이도)
def generate_low_difficulty_sentence():
    pattern = random.choice(low_value_patterns)
    place = random.choice(locations)
    place2 = random.choice([p for p in locations if p != place])
    object_ = random.choice(objects)
    phenomenon = random.choice(phenomena)
    task = random.choice(tasks)
    item = random.choice(borrow_items)

    sentence = pattern.format(
        place=place,
        place2=place2,
        object=object_,
        phenomenon=phenomenon,
        task=task,
        item=item
    )

    title = sentence
    content = f"{sentence}!"
    return {'요청글 제목': title, '요청글 내용': content, '난이도': '하'}



# ---------------------
# 5️⃣ 원본 데이터 불러오기
# ---------------------
file_path = '250825_price.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')
df['text'] = df['요청글 제목'].fillna('') + ' ' + df['요청글 내용'].fillna('')
df['난이도'] = df['text'].apply(classify_difficulty)

# ---------------------
# 6️⃣ 증강 대상 수 계산
# ---------------------
num_high_current = df['난이도'].value_counts().get('상', 0)
num_medium_current = df['난이도'].value_counts().get('중', 0)
num_low_current=df['난이도'].value_counts().get('하',0)
num_high_to_generate = max(0, 100000 - num_high_current)
num_medium_to_generate = max(0, 100000 - num_medium_current)
num_low_to_generate=max(0,100000-num_low_current)
augmented_list = []


if num_high_to_generate > 0:
    print(f"\n'상' 난이도 데이터 {num_high_to_generate}개를 증강합니다...")
    for _ in range(num_high_to_generate):
        # 아이템 오탈자 랜덤 적용
        item = random.choice(high_value_keywords)
        item_variant = random.choice(typo_variants_korean(item))
        sentence = generate_sentence([item_variant], '상')
        augmented_list.append(sentence)

if num_medium_to_generate > 0:
    print(f"'중' 난이도 데이터 {num_medium_to_generate}개를 증강합니다...")
    for _ in range(num_medium_to_generate):
        item = random.choice(medium_value_items)
        item_variant = random.choice(typo_variants_korean(item))
        sentence = generate_sentence([item_variant], '중')
        augmented_list.append(sentence)

if num_low_to_generate>0:
    print(f"'하' 난이도 데이터 {num_low_to_generate}개를 증강합니다...")
    for _ in range(num_low_to_generate):
        sentence=generate_low_difficulty_sentence()
        augmented_list.append(sentence)



if augmented_list:
    df_augmented = pd.DataFrame(augmented_list)
    df = pd.concat([df, df_augmented], ignore_index=True)
    print("데이터 병합 완료.")




def assign_price(diff):
    if diff=='하': return 500
    elif diff=='중': return 1500
    else: return 4000

df['기본가격'] = df['난이도'].apply(assign_price)
df['요청 시각'] = [random.randint(0, 23) for _ in range(len(df))]

def get_time_surcharge(hour):
    if 1 <= hour < 7: return 700
    elif 7 <= hour < 9: return 500
    elif 18 <= hour or hour < 1: return 300
    else: return 0

df['시간보상'] = df['요청 시각'].apply(get_time_surcharge)
df['날씨'] = random.choices(weather_conditions, weights=weather_probabilities, k=len(df))
df['날씨보상'] = df['날씨'].map(weather_bonus)
df['주말여부'] = [random.choice(['평일','주말']) for _ in range(len(df))]
df['주말보상'] = df['주말여부'].apply(lambda x: 300 if x=='주말' else 0)

def get_variation(diff):
    if diff=='하': return random.randint(-100,100)
    else: return random.randint(-200,200)

df['변동보상'] = df['난이도'].apply(get_variation)
numeric_cols = ['기본가격','시간보상','날씨보상','주말보상','변동보상']
df['최종가격'] = (df[numeric_cols].sum(axis=1)/100).round()*100

# ---------------------
# 11️⃣ 최종 저장
# ---------------------
print("\n[증강 후] 최종 데이터 분포:")
print(df['난이도'].value_counts())

df.to_excel("generated_training_data.xlsx", index=False)
print("\n✅ 학습용 데이터 증강 완료.")
