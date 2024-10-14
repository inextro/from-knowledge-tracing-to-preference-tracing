class User:
    def __init__(self, user_id, params):
        self.user_id = user_id
        self.params = params
        self.history = {}
        self._current_state = {}
        
        # preference state 및 history 초기화
        self._init_state()
    

    def _init_state(self):
        mcs = self.params[self.params['param'] == 'prior']['skill'].values
        priors = self.params[self.params['param'] == 'prior']['value'].values
        
        # 각 mc의 state를 prior로 초기화
        for mc, prior in zip(mcs, priors):
            self._current_state[mc] = prior

        # history에 0번 시점을 기록
        self.history[0] = self._current_state
    

    def get_pf_state(self):
        return self._current_state
    

    def get_history(self, t=None):
        if t is not None:
            return self.history[t]
        
        else:
            return self.history


    def compute_posterior(self, movie_id, mc, response):
        prior = self.get_pf_state()[mc]
        learn = self.params[
            (self.params['skill'] == mc) & 
            (self.params['param'] == 'learns')
            ]['value'].values[0]
        guess = self.params[
            (self.params['skill'] == mc) & 
            (self.params['param'] == 'guesses') & 
            (self.params['class'] == str(movie_id))
            ]['value'].values[0]
        slip = self.params[
            (self.params['skill'] == mc) & 
            (self.params['param'] == 'slips') & 
            (self.params['class'] == str(movie_id))
            ]['value'].values[0]
        
        if response: # like
            correct = prior * (1 - slip) + (1 - prior) * guess
            observed_mastery = (prior * (1 - slip)) / (correct + 1e-1000)

        else: # dislike
            incorrect = prior * slip + (1 - prior) * (1 -guess)
            observed_mastery = (prior * slip) / (incorrect + 1e-1000)
        
        updated_mastery = observed_mastery + (1 - observed_mastery) * learn
        
        return updated_mastery
    

    def update_state(self, movie_ids, mcs, responses):
        # preference state 업데이트
        for movie_id, mc, response in zip(movie_ids, mcs, responses):
            updated_mastery = self.compute_posterior(movie_id=movie_id, mc=mc, response=response)
            updated_state = self._current_state.copy()
            updated_state[mc] = updated_mastery

            self._current_state = updated_state
        
            # history 기록
            new_t = max(self.history.keys()) + 1
            self.update_history(new_t)

    def update_history(self, t):
        self.history[t] = self._current_state