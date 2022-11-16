while True:
    exercise = input('운동을 입력하세요 : ')
    print(exercise)

    if exercise == 'x':
        break
    import sangche
    
    if exercise == '어깨':
        exec(open('shoulder_press.py', encoding='UTF-8').read())
        print('어깨 운동 시작')
        
    elif exercise == '팔':
        exec(open('single_armcurl.py', encoding='UTF-8').read())
        print('팔 운동 시작')