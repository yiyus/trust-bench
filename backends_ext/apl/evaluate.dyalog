∇res←Evaluate req;f;hf;r;j;h;parsed;family;param
  f←NameFor req.problem_id
  :If 0≠≢f
      hf←HessianNameFor req.problem_id
      r j←Apply f(req.x)
      h←Apply hf(req.x)
  :Else
      parsed←ParseParametrised req.problem_id
      :If 0=≢parsed
          res←ErrorResult'Unknown problem_id: ',req.problem_id
          :Return
      :EndIf
      family param←parsed
      f←FamilyNameFor family
      :If 0=≢f
          res←ErrorResult'Unknown problem_id: ',req.problem_id
          :Return
      :EndIf
      hf←FamilyHessianNameFor family
      r j←ApplyParam f param(req.x)
      h←ApplyParam hf param(req.x)
  :EndIf
  res←⎕NS''
  res.problem_id←req.problem_id
  res.status←'OK'
  res.message←NULL
  res.residual←r
  res.jacobian←j
  res.hessian←h
∇
