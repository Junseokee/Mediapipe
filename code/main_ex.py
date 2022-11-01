while True:
    exercise = input('운동을 입력하세요 : ')
    print(exercise)

    if exercise == 'x':
        break
    import sangche
    
    if exercise == '어깨':
        sangche.shoulder()
        print('어깨 운동 시작')
        
    elif exercise == '팔':
        sangche.arm()
        print('팔 운동 시작')