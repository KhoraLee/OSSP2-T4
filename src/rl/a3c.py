from .learner import ReinforcementLearner
from . import A2CLearner
from networks import Network, DNN, LSTMNetwork, CNN
import threading

class A3CLearner(ReinforcementLearner):
    def __init__(self, *args, list_stock_code=None,
        list_chart_data=None, list_training_data=None,
        list_min_trading_price=None, list_max_trading_price=None,
        value_network_path=None, policy_network_path=None,
        **kwargs):
        assert len(list_training_data) > 0
        super().__init__(*args, **kwargs)
        self.num_features += list_training_data[0].shape[1]

        # 공유 신경망 생성
        self.shared_network = Network.get_shared_network(
            net=self.net, num_steps=self.num_steps,
            input_dim=self.num_features,
            output_dim=self.agent.NUM_ACTIONS)
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)

        # A2CLearner 생성
        self.learners = []
        for (stock_code, chart_data, training_data,
            min_trading_price, max_trading_price) in zip(
                list_stock_code, list_chart_data, list_training_data,
                list_min_trading_price, list_max_trading_price
            ):
            learner = A2CLearner(*args,
                stock_code=stock_code, chart_data=chart_data,
                training_data=training_data,
                min_trading_price=min_trading_price,
                max_trading_price=max_trading_price,
                shared_network=self.shared_network,
                value_network=self.value_network,
                policy_network=self.policy_network, **kwargs)
            self.learners.append(learner)

    def run(self, learning=True):
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(
                target=learner.run, daemon=True, kwargs={'learning': learning}
            ))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def predict(self):
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(
                target=learner.predict, daemon=True
            ))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()