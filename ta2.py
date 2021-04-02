import numpy as np
import math
import talib as ta

def IntPortion(param):
   if (param > 0): return math.floor(param)
   if (param < 0): return math.ceil(param)
   return 0.0

def jma(input, Length, Phase):

    closeLength = len(input)

    JMAValueBuffer = [0.0] * closeLength
    fC0Buffer = [0.0] * closeLength
    fA8Buffer = [0.0] * closeLength
    fC8Buffer = [0.0] * closeLength

    # temporary buffers
    list = [0.0] * 128
    ring1 = [0.0] * 128
    ring2 = [0.0] * 11
    buffer = [0.0] * 62

    # int vars
    initFlag = False
    limitValue = 0
    startValue = 0
    loopParam = 0
    loopCriteria = 0
    cycleLimit = 0
    highLimit = 0
    counterA = 0
    counterB = 0

    # double vars
    cycleDelta = 0.0
    lowDValue = 0.0
    highDValue = 0.0
    absValue = 0.0
    paramA = 0.0
    paramB = 0.0
    phaseParam = 0.0
    logParam = 0.0
    JMAValue = 0.0
    series = 0.0
    sValue = 0.0
    sqrtParam = 0.0
    lengthDivider = 0.0

    # temporary int variables
    s58 = 0
    s60 = 0
    s40 = 0
    s38 = 0
    s68 = 0

    # init ---------------------------------------------------------------------------------------------------

    lengthParam = 0.0
    limitValue = 63
    startValue = 64
    for i in range(limitValue+1): list[i] = -1000000
    for i in range(startValue,128): list[i] = 1000000

    initFlag = True
    if (Length < 1.0000000002): lengthParam = 0.0000000001
    else: lengthParam = (Length - 1) / 2.0

    if (Phase < -100): phaseParam = 0.5
    elif (Phase > 100): phaseParam = 2.5
    else: phaseParam = Phase / 100.0 + 1.5

    logParam =  math.log(np.sqrt(lengthParam)) / math.log (2.0)

    if (logParam + 2.0 < 0): logParam = 0
    else: logParam = logParam + 2.0

    sqrtParam = np.sqrt(lengthParam) * logParam
    lengthParam   = lengthParam * 0.9
    lengthDivider = lengthParam / (lengthParam + 2.0)

    # JMA compute ----------------------------------------------------------------------------------------------------------------------

    Bars = len(input)
    counted_bars = 0
    limit = Bars - counted_bars - 1

    # main cycle
        
    for shift in range(limit,-1,-1):
        series = input[-shift-1]
        if (loopParam < 61):
            loopParam += 1
            buffer[loopParam] = series
        if (loopParam > 30):
            if (initFlag):
                initFlag = False
                diffFlag = 0
                for i in range(1,30):
                    if (buffer[i + 1] != buffer[i]): diffFlag = 1
                highLimit = diffFlag * 30
             
                if (highLimit == 0): paramB = series
                else: paramB = buffer[1]
             
                paramA = paramB
                if (highLimit > 29): highLimit = 29
            else: 
                highLimit = 0

            # big cycle
                                    
            for i in range(highLimit,-1,-1):
                if (i == 0): sValue = series
                else: sValue = buffer [31 - i]

                if (abs(sValue - paramA) > abs (sValue - paramB)): absValue = abs(sValue - paramA)
                else: absValue = abs(sValue - paramB)
                dValue = absValue + 0.0000000001 #1.0e-10 

                if (counterA <= 1): counterA = 127
                else: counterA -= 1

                if (counterB <= 1): counterB = 10
                else: counterB -= 1 

                if (cycleLimit < 128): cycleLimit+=1 
                cycleDelta += (dValue - ring2[counterB]) 
                ring2[counterB] = dValue 
                if (cycleLimit > 10): highDValue = cycleDelta / 10.0
                else: highDValue = cycleDelta / cycleLimit

                if (cycleLimit > 127): 
                    dValue = ring1[counterA]
                    ring1[counterA] = highDValue 
                    s68 = 64
                    s58 = s68
                    while (s68 > 1):
                        if (list[s58] < dValue):
                            s68 = s68 // 2
                            s58 += s68
                        else: 
                            if (list[s58] <= dValue): 
                                s68 = 1
                            else:  
                                s68 = s68 // 2
                                s58 -= s68
                else:
                    ring1 [counterA] = highDValue 
                    if ((limitValue + startValue) > 127):
                        startValue -= 1
                        s58 = startValue
                    else:
                        limitValue +=1 
                        s58 = limitValue 

                    if (limitValue > 96): s38 = 96
                    else: s38 = limitValue 

                    if (startValue < 32): s40 = 32
                    else: s40 = startValue 

                s68 = 64
                s60 = s68
                while (s68 > 1):
                    if (list[s60] >= highDValue):
                        if (list [s60 - 1] <= highDValue):
                            s68 = 1
                        else:
                            s68 = s68 // 2
                            s60 -= s68 
                    else:
                        s68 = s68 // 2
                        s60 += s68
                    if ((s60 == 127) and (highDValue > list[127])): s60 = 128 

                if (cycleLimit > 127):
                    if (s58 >= s60):
                        if (((s38 + 1) > s60) and ((s40 - 1) < s60)): 
                            lowDValue += highDValue
                        elif ((s40 > s60) and ((s40 - 1) < s58)): 
                            lowDValue += list [s40 - 1]
                    elif (s40 >= s60):
                        if (((s38 + 1) < s60) and ((s38 + 1) > s58)): 
                            lowDValue += list[s38 + 1]
                    elif ((s38 + 2) > s60): 
                        lowDValue += highDValue 
                    elif (((s38 + 1) < s60) and ((s38 + 1) > s58)): 
                        lowDValue += list[s38 + 1] 

                    if (s58 > s60):
                        if (((s40 - 1) < s58) and ((s38 + 1) > s58)): 
                            lowDValue -= list [s58] 
                        elif ((s38 < s58) and ((s38 + 1) > s60)): 
                            lowDValue -= list[s38] 
                    else:
                        if (((s38 + 1) > s58) and ((s40 - 1) < s58)): 
                            lowDValue -= list[s58]
                        elif ((s40 > s58) and (s40 < s60)): 
                            lowDValue -= list[s40] 

                if (s58 <= s60):
                    if (s58 >= s60): list[s60] = highDValue
                    else:
                        for j in range(s58+1, s60):
                            list[j - 1] = list[j] 
                        list[s60 - 1] = highDValue 
                else:
                    for j in range(s58-1,s60-1,-1):
                        list [j + 1] = list [j]
                    list[s60] = highDValue 

                if (cycleLimit <= 127):
                    lowDValue = 0 
                    for j in range(s40, s38+1):
                        lowDValue += list[j] 

                # ---

                if ((loopCriteria + 1) > 31): loopCriteria = 31
                else: loopCriteria += 1
                        
                JMATempValue = 0.0
                sqrtDivider = sqrtParam / (sqrtParam + 1.0)

                if (loopCriteria <= 30):
                    if (sValue - paramA > 0): paramA = sValue
                    else: paramA = sValue - (sValue - paramA) * sqrtDivider 

                    if (sValue - paramB < 0): paramB = sValue
                    else: paramB = sValue - (sValue - paramB) * sqrtDivider 

                    JMATempValue = series

                    if (loopCriteria == 30):
                        fC0Buffer [shift] = series
                        intPart = 0

                        if (math.ceil(sqrtParam) >= 1): intPart = math.ceil(sqrtParam)
                        else: intPart = 1
                        leftInt = IntPortion (intPart) 
                        if (math.floor(sqrtParam) >= 1): intPart = math.floor(sqrtParam) 
                        else: intPart = 1
                        rightPart = IntPortion (intPart)

                        if (leftInt == rightPart): dValue = 1.0
                        else: dValue = (sqrtParam - rightPart) / (leftInt - rightPart)

                        upShift = 0
                        if (rightPart <= 29): upShift = rightPart
                        else: upShift = 29
                                
                        dnShift = 0
                        if (leftInt <= 29): dnShift = leftInt
                        else: dnShift = 29
                                
                        fA8Buffer[shift] = (series - buffer [loopParam - upShift]) * (1 - dValue) / rightPart + (series - buffer[loopParam - dnShift]) * dValue / leftInt
                else:
                    powerValue = 0.0
                    squareValue = 0.0
                    dValue = lowDValue / (s38 - s40 + 1)
                    if (0.5 <= logParam - 2.0): powerValue = logParam - 2.0
                    else: powerValue = 0.5
               
                    if (logParam >= math.pow(absValue/dValue, powerValue)): dValue = math.pow(absValue/dValue, powerValue)
                    else: dValue = logParam 

                    if (dValue < 1): dValue = 1

                    powerValue = math.pow(sqrtDivider, math.sqrt(dValue)) 
                    if (sValue - paramA > 0): paramA = sValue
                    else: paramA = sValue - (sValue - paramA) * powerValue 
                    if (sValue - paramB < 0): paramB = sValue
                    else: paramB = sValue - (sValue - paramB) * powerValue

            # end of big cycle                  			   
                
            if (loopCriteria > 30):
                JMATempValue = JMAValueBuffer[shift + 1]
                powerValue   = math.pow(lengthDivider, dValue)
                squareValue  = math.pow(powerValue, 2)
                         
                fC0Buffer[shift] = (1 - powerValue) * series + powerValue * fC0Buffer[shift + 1]
                fC8Buffer[shift] = (series - fC0Buffer[shift]) * (1 - lengthDivider) + lengthDivider * fC8Buffer[shift + 1]
                fA8Buffer[shift] = (phaseParam * fC8Buffer[shift] + fC0Buffer[shift] - JMATempValue) * (powerValue * (-2.0) + squareValue + 1) + squareValue * fA8Buffer [shift + 1]  
                JMATempValue += fA8Buffer [shift] 

            JMAValue = JMATempValue

        if (loopParam <= 30): JMAValue = 0
        JMAValueBuffer[shift] = JMAValue
    # End of main cycle
    JMAValueBuffer.reverse()
    JMAValueBuffer[0:30] = [np.nan] *30
    return np.array(JMAValueBuffer)

def jmacd(input, slow_period, fast_period, phase):
    slow = jma(input, slow_period, phase)
    fast = jma(input, fast_period, phase)
    return fast - slow

def jmamom(input, jma_period, jma_phase, mom_period):
    return ta.MOM(jma(input, jma_period, jma_phase), mom_period)





