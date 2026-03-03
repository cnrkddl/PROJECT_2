import google.generativeai as genai
genai.configure(api_key="AIzaSyAb0-rAFxZI_OPP5PQxuD-zrSLQPg-4y3s")
for m in genai.list_models():
    print(m.name)
