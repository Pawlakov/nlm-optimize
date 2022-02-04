module NLMImplemFSharp.Implementation

let fiL2FloatDist (u0:list<float32>) (u1:list<float32>) i0 j0 i1 j1 radius channels width0 width1 =

    let fiL2FloatDistInner (uu0:list<float32>) (uu1:list<float32>) ii =
        
        if ii < channels then
            let rec outerLoop s =
                if s <= radius then
                    let ptr0 = ((j0 + s) * width0) + (i0 - radius)
                    let ptr1 = ((j1 + s) * width1) + (i1 - radius)
                        
                    let rec innerLoop r ptr1 =
                        if r <= radius then
                            let dif = uu0[ptr0] - uu1[ptr1]
                            let innerDist = dif * dif
                            let innerResult = innerLoop (r + 1) (ptr1 + 1)
                            innerResult + innerDist
                        else
                            0.0f
        
                    let innerLoopResult = innerLoop (-radius) ptr1
                    let outerResult = outerLoop (s + 1)
                    innerLoopResult + outerResult
                else
                    0.0f
           
            let result1 = outerLoop (-radius)
            let result2 = fiL2FloatDistInner u0[ii + 1] u1[ii + 1] (ii + 1)
            result1 + result2
        else
            0.0f

    let ii = 0
    let dif = fiL2FloatDistInner u0[ii] u1[ii] ii
    dif

let denoise iDWin iDBloc fSigma fFiltPar fpI fpO iChannels iWidth iHeight =
    printfn "Hello %s" name
