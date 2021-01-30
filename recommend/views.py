from django.shortcuts import render
import pandas as pd

from .models import jobdict,recruit_info
import os
import warnings
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')


def index (request):
    context = {
        '언어': {
            'id': 'Languages',
            'skills':
                [
                    ['Java', 'Java'],
                    ['Python', 'Python'],
                    ['C', 'Clang'],
                    ['C#', 'CS'],
                    ['C++', 'Cpp'],
                    ['Go', 'Go'],
                    ['Kotlin', 'Kotlin'],
                    ['JavaScript', 'JavaScript'],
                    ['HTML', 'HTML'],
                    ['CSS', 'CSS'],
                    ['PHP', 'PHP'],
                    ['JSP', 'JSP'],
                    ['TypeScript', 'TypeScript'],
                    ['Rupy', 'Rupy'],
                    ['Scala', 'Scala'],
                ]
        },
        '웹개발': {
            'id': 'WEB',
            'skills':
                [
                    ['Spring', 'Spring'],
                    ['Django', 'Django'],
                    ['Vue.js', 'Vue'],
                    ['Angular.js', 'Angular'],
                    ['.NET', 'DotNet'],
                    ['ASP.NET', 'ASPDot'],
                    ['Rails', 'Rails'],
                    ['jQuery', 'JQuery'],
                    ['React.js', 'React'],
                    ['Redux', 'Redux'],
                    ['Node.js', 'Node'],
                    ['ASP', 'ASP'],
                    ['Webpack', 'Webpack'],
                    ['Ajax', 'Ajax'],
                ]
        },
        '앱개발': {
            'id': 'APPS',
            'skills': [
                ['IOS', 'IOS'],
                ['Android', 'Android'],
                ['Swift', 'Swift'],
                ['React Native', 'React Native'],
            ]
        },
        'DB': {
            'id': 'DB',
            'skills': [
                ['Oracle', 'Oracle'],
                ['MySQL', 'MySQL'],
                ['MsSQL', 'MsSQL'],
                ['NoSQL', 'NoSQL'],
                ['MariaDB', 'MariaDB'],
                ['PostgreSQL', 'PostgreSQL'],
                ['MongoDB', 'MongoDB'],
                ['Redis', 'Redis'],
                ['GraphQL', 'GraphQL'],
            ]
        },
        'AI': {
            'id': 'AI',
            'skills': [
                ['Machine Learning', 'ML'],
                ['Deep Learning', 'DL'],
                ['CV', 'CV'],
                ['PyTorch', 'Pytorch'],
                ['TensorFlow', 'Tensor'],
            ]
        },
        '빅데이터_서버': {
            'id': 'BigSer',
            'skills': [
                ['Linux', 'Linux'],
                ['Docker', 'Docker'],
                ['Kubernetes', 'Kubernetes'],
                ['Hadoop', 'Hadoop'],
                ['Spark', 'Spark'],
                ['Zeplin', 'Zeplin'],
                ['Kafka', 'Kafka'],
                ['WAS', 'WAS'],
                ['AWS', 'AWS'],
                ['GCP', 'GCP'],
                ['Azure', 'Azure'],
            ]
        },
        '협업 툴': {
            'id': 'Coll',
            'skills': [
                ['Git', 'Git'],
                ['JIRA', 'JIRA'],
                ['Jenkins', 'Jenkins'],
                ['Slack', 'Slack'],
            ]
        },
        '기타': {
            'id': 'ETC',
            'skills': [
                ['Photoshop', 'Photoshop'],
                ['Sketch', 'Sketch'],
                ['Illustrator', 'Illustrator'],
                ['Unity3D', 'Unity3D'],
                ['Maya', 'Maya'],
            ]
        }
    }

    return render(request,'index.html',context)
#@@@@@@@@@@@@@@@@@@@@
def fir_result(request):
    context={}

    df=pd.DataFrame(list(jobdict.objects.all().values()))
    df=df.dropna(axis=0)
    #context={'df':df}


    df['job'][4]='QA'
    df['job'][10]='DBA'

    vectorizer=CountVectorizer()  # 피쳐 벡터화

    feature_vector_job=vectorizer.fit_transform(df['skill']) #피쳐 벡터 행렬
    feature_job=vectorizer.get_feature_names()


    person_vec=np.zeros((1,70))
    #person 임의로 줌 input값!!
    #person=['vue닷js','redux','ruby','닷net']
    person=['python']
    for skill in person:
        index=feature_job.index(skill)
        person_vec[0][index]=1
    person_vec

    # 코사인 유사도 계산

    skill_sim=cosine_similarity(feature_vector_job,person_vec)

    df['similarity']=''
    df['similarity']=skill_sim
    df.sort_values(by='similarity', ascending=False).head(10)

    job_sim_result=df.sort_values(by='similarity', ascending=False).head(3)
    job_sim_result=job_sim_result['job'].values


    # 공고 추천

    # total 공고 파일 업로드
    df_total= pd.DataFrame(list(recruit_info.objects.all().values()))

    df_total_new=df_total.copy()

    df_total_new=df_total_new.dropna(axis=0,subset=["total_skill_literal"])


    vectorizer=CountVectorizer()  # 피쳐 벡터화

    feature_vector_title=vectorizer.fit_transform(df_total_new['total_skill_literal']) #피쳐 벡터 행렬
    feature_title=vectorizer.get_feature_names()


    person_vec_title=np.zeros((1,70))

    for skill in person:
        index=feature_title.index(skill)
        person_vec[0][index]=1
    person_vec

    # 코사인 유사도 계산

    skill_sim=cosine_similarity(feature_vector_title,person_vec)

    df_total_new['similarity']=''
    df_total_new['similarity']=skill_sim
    df_total_new_sorting=df_total_new.sort_values(by='similarity', ascending=False)
    df_total_new_sorting[:]



    # 나온 공고중에 직무를 만족시키는 공고 5개 뽑기

    job_1=job_sim_result[0]
    df_final_1=df_total_new_sorting[df_total_new_sorting.job.str.find(job_1)>-1]

    job_2=job_sim_result[1]
    df_final_2=df_total_new_sorting[df_total_new_sorting.job.str.find(job_2)>-1]

    job_3=job_sim_result[2]
    df_final_3=df_total_new_sorting[df_total_new_sorting.job.str.find(job_3)>-1]
    context={'job1':job_1, 'df':df_final_1, 'job2':job_2, 22:df_final_2, 'job3':job_3, 33:df_final_3}
    return render(request,'result1.html',context)

def wanna_job(request):
    context2 = {'job': [
        {'id': 'web', 'name': '웹개발'}, {'id': 'network', 'name': '네트워크/보안/운영'}, {'id': 'se', 'name': '시스템엔지니어'},
        {'id':'soft','name':'소프트웨어엔지니어'},{'id': 'qa', 'name': 'Quality Assurance'},
        {'id': 'plan', 'name': '기획'}, {'id': 'da', 'name': '데이터분석'}, {'id': 'app', 'name': '모바일앱개발'},
        {'id': 'pm','name':'프로젝트매니저'}, {'id': 'game', 'name': '게임개발'},
        {'id': 'dba', 'name': 'Database Admin'}, {'id': 'consulting', 'name': '컨설팅'}, {'id': 'he', 'name': '하드웨어엔지니어'},
        {'id': 'cd', 'name': '캐릭터디자인'},
        {'id': 'gd', 'name': '그래픽디자인'}, {'id': 'ad', 'name': '애니메이션디자인'}, {'id': 'marketing', 'name': '마케팅'},
        {'id': 'market', 'name': '시장조사/분석'},
        {'id': 'sa', 'name': '소프트웨어아키텍트'}, {'id': 'wp', 'name': '웹퍼블리셔'}, {'id': 'ud', 'name': 'UI/UX/GUI디자인'},
        {'id': 'md', 'name': '상품개발/기획/MD'},
        {'id': 'strategy', 'name': '경영기획/전략'}, {'id': 'advertise', 'name': '광고/시각디자인'},
        {'id': 'video', 'name': '영상/모션디자인'}, {'id': 'sales', 'name': '영업기획/관리/지원'},
        {'id': 'support', 'name': '경영지원'}, {'id': 'erp', 'name': 'ERP'}, {'id': 'iu', 'name': '일러스트레이터'},
        {'id': 'wd', 'name': '웹디자인'}, {'id': 'machine', 'name': '기계'},
        {'id': 'esemi', 'name': '전자/반도체'}, {'id': 'control', 'name': '제어'}, {'id': 'elec', 'name': '전기'},
        {'id': 'dsemi', 'name': '반도체/디스플레이'},
        {'id': 'econtrol', 'name': '전기/전자/제어'}, {'id': 'domsales', 'name': '국내영업'},
        {'id': 'sound', 'name': '음악/음향/사운드'}, {'id': 'itsol', 'name': 'IT/솔루션영업'},
        {'id': 'emarketing', 'name': '온라인마케팅'}, {'id': 'smarketing', 'name': '전략마케팅'}, {'id': 'bd', 'name': '브랜드디자인'},
        {'id': 'bmarketing', 'name': '브랜드마케팅'},
        {'id': 'publish', 'name': '출판/편집디자인'}, {'id': 'pd', 'name': '방송연출/PD/감독'}, {'id': 'cs', 'name': '고객지원/CS'},
        {'id': 'culture', 'name': '조직문화'},
        {'id': 'crm', 'name': 'CRM'}, {'id': 'maintenance', 'name': '유지/수리/정비'},{'id':'tech','name':'기술영업'}, {'id': 'product', 'name': '제품/산업디자인'}]}
    return render(request,'infojob.html',context2)

def sec_result(requese):
    #  파일 로드 및 불필요한 스킬 제거
    job=['web','se']#사용자가 선택한 잡 리스트 html에서 가져오기
    skillPerJob = pd.read_csv('skillPerJob.csv', index_col="index")
    skillPerJob = skillPerJob.drop(['WEB', 'App', 'DB', 'Data'])

    # 최다 언급 skill 기준 정렬
    df = pd.DataFrame(skillPerJob[job])
    df = df.drop(df[df[job] == 0].index)
    df = df.sort_values(by=job, ascending=False)
    print("공고에서 언급한 기술 갯수 : ", df.shape[0])

    # 각 기술별 비율 컬럼 추가
    df_sum = df[job].sum()
    df['비율'] = df.apply(lambda x: round(x / df_sum * 100, 2))

    # 기술 + 비율 딕셔너리화 : 상위 10개 추출
    df = df.head(10)
    top_skil = dict(zip(list(df.index), df['비율'].to_list()))
    context3={'top_skill':top_skil}
    return render(requese,'result2.html',context3)
def readCsv(address):
    df=pd.read_csv(address)
    return df
# address 파라미터 적어줄때  => r'절대주소'

test_df=readCsv(r'C:\Users\네리\Desktop\sba_re\4조\job_data_list_final.csv')
print(test_df.columns)
#col=index,jobId,company,title,job,skill,region,experience,intro
#task,require,prefer,jobUrl
test_df=test_df.fillna('')

print(type(test_df['jobId'][0]))

#test => titleId,company,title,job,skill
def create():
    for i in range(len(test_df)):
        test.objects.create(titleId=test_df['jobId'][i], company=test_df['company'][i], title=test_df['title'][i], job=test_df['job'][i], skill=test_df['skill'][i])
print(len(test_df))
#model.objects.all()
#print(jobdict.objects.all())'
#django.core.exceptions.ImproperlyConfigured: Requested setting INSTALLED_APPS, but settings are not configured.
# You must either define the environment variable DJANGO_SETTINGS_MODULE or call settings.configure() before accessing settings.
