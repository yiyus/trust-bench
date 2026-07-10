Rosenbrock←{p q←⍵ ⋄ ((10×q-×⍨p),1-p)[(-20×p)10 ⋄ ¯1 0]}
RosenbrockHessian←{
    r j←Rosenbrock ⍵
    h1←2 2⍴¯20 0 0 0
    ((⍉j)+.×j)+r[1]×h1
}
