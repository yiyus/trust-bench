∇res←Evaluate req;f;hf;r;j;h
  f←NameFor req.problem_id
  :If 0=≢f
      res←ErrorResult'Unknown problem_id: ',req.problem_id
      :Return
  :EndIf
  hf←HessianNameFor req.problem_id
  r j←Apply f(req.x)
  h←Apply hf(req.x)
  res←⎕NS''
  res.problem_id←req.problem_id
  res.status←'OK'
  res.message←NULL
  res.residual←r
  res.jacobian←j
  res.hessian←h
∇
