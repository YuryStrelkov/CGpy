from cgeo.numeric_methods import rand_in_range
from typing import Tuple
import numpy as np
import warnings


warnings.filterwarnings('ignore')
_debug_mode = False

"""
Пусть есть два события связаны соотношением:
P{y=1|X} = f(z) (1)
P{y=0|X} = 1 - f(z) (2)
z = b + (X,T)
f(z)  = 1 / (1 + exp{-z})
d/dz f(z)  = z' * exp{-z} / (1 + exp{-z})^2 = z' * (1 - 1/f(z))/f(z)^2 = 
           = z' * ((1 - f(z)) * f(z)) (3)

Тогда соотношения (1) и (2):
P{y=1|X} = f(b + (X,T)) (4)
P{y=0|X} = 1 - f(b + (X,T)) (5)
Вероятность y при условии Х
P{y|X} = f(b + (X,T))^y*(1 - f(b + (X,T)))^(y-1) (6)
Условие максимального правдоподобия:
{b,T} = argmax(П P{y_i|X_i}) = argmax(? ln(P{y_i|X_i}))
argmax(? ln(P{y_i|X_i})) = ?y_i * ln(f(b + (X,T))) + (1-y_i)*(ln(1 - f(b + (X,T))))
требуеся найти производные для:
d/db argmax(? ln(P{y_i|X_i}))
d/t_j argmax(? ln(P{y_i|X_i})), где t_j элемент вектора T
Для этого распишем необходимые нам формулы:
d/dx ln(f(x)) = af'(x)/f(x)

d/db   ln(f(b + (X,T))) =       f'(b + (X,T))/f(b + (X,T)) =        1 - f(b + (X,T) 
d/dx_j ln(f(b + (X,T))) = d/dx_j f(b + (X,T))/f(b + (X,T)) = x_j * (1 - f(b + (X,T))
"""


def ellipsoid(x: float, y: float, params: Tuple[float, float, float, float, float]) -> float:
    """
    :param x:
    :param y:
    :param params:
    :return:
    """
    return x * params[0] + y * params[1] + x * y * params[2] + x * x * params[3] + y * y * params[4] - 1


def log_reg_ellipsoid_test_data(params: Tuple[float, float, float, float, float],
                                arg_range: float = 5.0, rand_range: float = 1.0,
                                n_points: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param params:
    :param arg_range:
    :param rand_range:
    :param n_points:
    :return:
    """
    if _debug_mode:
        print(f"logistic regression f(x,y) = {params[0]:1.3}x + {params[1]:1.3}y + {params[2]:1.3}xy +"
              f" {params[3]:1.3}x^2 + {params[4]:1.3}y^2 - 1,\n"
              f" arg_range =  [{-arg_range * 0.5:1.3}, {arg_range * 0.5:1.3}],\n"
              f" rand_range = [{-rand_range * 0.5:1.3}, {rand_range * 0.5:1.3}]")
    features = np.zeros((n_points, 5), dtype=float)
    features[:, 0] = np.array([rand_in_range(arg_range) for _ in range(n_points)])
    features[:, 1] = np.array([rand_in_range(arg_range) for _ in range(n_points)])
    features[:, 2] = features[:, 0] * features[:, 1]
    features[:, 3] = features[:, 0] * features[:, 0]
    features[:, 4] = features[:, 1] * features[:, 1]
    groups =  np.array(
        [np.sign(ellipsoid(features[i, 0], features[i, 1], params)) * 0.5 + 0.5 for i in range(n_points)])
    return features, groups


def log_reg_test_data(k: float = -1.5, b: float = 0.1, arg_range: float = 1.0,
                      rand_range: float = 0.0, n_points: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param k:
    :param b:
    :param arg_range:
    :param rand_range:
    :param n_points:
    :return:
    """
    if _debug_mode:
        print(f"logistic regression test data b = {b:1.3}, k = {k:1.3},\n"
              f" arg_range = [{-arg_range * 0.5:1.3}, {arg_range * 0.5:1.3}],\n"
              f" rand_range = [{-rand_range * 0.5:1.3}, {rand_range * 0.5:1.3}]")
    features = np.zeros((n_points, 2), dtype=float)
    features[:, 0] = np.array([rand_in_range(arg_range) for _ in range(n_points)])
    features[:, 1] = np.array([rand_in_range(arg_range) for _ in range(n_points)])
    groups = np.array(
        [1 if features[i, 0] * k + b > features[i, 1] + rand_in_range(rand_range) else 0.0 for i in range(n_points)])
    return features, groups


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Сигмоид, хе-хе
    :param x:
    :return:
    """
    return 1.0 / (1.0 + np.exp(-x))


def loss(groups_probs, groups) -> float:
    """

    :param groups_probs:
    :param groups:
    :return:
    """
    return (-groups * np.log(groups_probs) - (1.0 - groups) * np.log(1.0 - groups_probs)).mean()


"""
def draw_logistic_data(features: np.ndarray, groups: np.ndarray, theta: np.ndarray = None) -> None:
    [plt.plot(features[i, 0], features[i, 1], '+b') if groups[i] == 0
     else plt.plot(features[i, 0], features[i, 1], '*r') for i in
     range(features.shape[0] // 2)]

    if theta is None:
        plt.show()
        return

    b = theta[0] / np.abs(theta[2])
    k = theta[1] / np.abs(theta[2])

    x_0, x_1 = features[:, 0].min(), features[:, 0].max()
    y_0, y_1 = features[:, 1].min(), features[:, 1].max()

    y_1 = (y_1 - b) / k
    y_0 = (y_0 - b) / k

    if y_0 < y_1:
        x_1 = min(x_1, y_1)
        x_0 = max(x_0, y_0)
    else:
        x_1 = min(x_1, y_0)
        x_0 = max(x_0, y_1)

    x = [x_0, x_1]
    y = [b + x_0 * k, b + x_1 * k]
    plt.plot(x, y, 'k')
    plt.show()
"""


class LogisticRegression:
    """

    """
    def __init__(self, learning_rate: float = 1.0, max_iters: int = 1000, accuracy: float = 1e-2):
        """

        :param learning_rate:
        :param max_iters:
        :param accuracy:
        """
        self.__max_train_iters: int = 0
        self.__learning_rate: float = 0
        self.__learning_accuracy: float = 0
        self.__group_features_count: int = 0
        self.__losses: float = 0.0
        self.__thetas: np.ndarray

        self.max_train_iters = max_iters
        self.learning_rate = learning_rate
        self.learning_accuracy = accuracy

    def __str__(self):
        return f"{{\n" \
               f"\t\"group_features_count\" : {self.group_features_count},\n" \
               f"\t\"max_train_iters\":       {self.max_train_iters},\n" \
               f"\t\"learning_rate\":         {self.learning_rate},\n" \
               f"\t\"learning_accuracy\":     {self.learning_accuracy},\n" \
               f"\t\"thetas\":                [{', '.join(str(e) for e in self.thetas.flat)}],\n" \
               f"\t\"losses\":                {self.losses}\n" \
               f"}}"

    def __call__(self, *args):
        if len(args) != 2:
            return
        x, group = args
        self.train(x, group)

    @property
    def group_features_count(self) -> int:
        return self.__group_features_count

    @property
    def max_train_iters(self) -> int:
        return self.__max_train_iters

    @max_train_iters.setter
    def max_train_iters(self, value: int) -> None:
        self.__max_train_iters = min(max(value, 100), 100000)

    @property
    def learning_rate(self) -> float:
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.__learning_rate = min(max(value, 0.001), 1.0)

    @property
    def learning_accuracy(self) -> float:
        return self.__learning_accuracy

    @learning_accuracy.setter
    def learning_accuracy(self, value: float) -> None:
        self.__learning_accuracy = min(max(value, 0.001), 1.0)

    @property
    def thetas(self) -> np.ndarray:
        return self.__thetas

    @property
    def losses(self) -> float:
        return self.__losses

    def predict(self, features: np.ndarray):
        if features.ndim != 2:
            print("wrong predict features data")
            return -1.0
        if features.shape[1] != self.thetas.size - 1:
            print("wrong predict features data")
            return -1.0
        return sigmoid(features @ self.thetas[1::] + self.thetas[0])

    def train(self, features: np.ndarray, groups: np.ndarray):
        if features.ndim != 2:
            print("wrong predict features data")
            return -1.0
        self.__group_features_count = features.shape[1]
        self.__thetas: np.ndarray = np.array([rand_in_range(1000) for _ in range(self.__group_features_count + 1)])
        x = np.hstack((np.ones((features.shape[0], 1), dtype=float), features[:, 0: self.group_features_count]))
        thetas: np.ndarray = self.thetas.copy()
        iteration = 0
        while True:
            self.__thetas = thetas - self.learning_rate * (x.T @ (sigmoid(x @ thetas) - groups))
            iteration += 1
            if iteration >= self.max_train_iters:
                if _debug_mode:
                    print(f"trainings stopped after exceed available iterations count : {self.max_train_iters}")
                break
            if np.sqrt(np.power(thetas - self.thetas, 2.0).sum()) <= self.learning_accuracy:
                if _debug_mode:
                    print(f"trainings stopped after satisfy accuracy constraints.\n"
                          f"Eps: {self.learning_accuracy}, Iters: {iteration}")
                break
            thetas = self.thetas.copy()
        self.__losses = loss(self.predict(features), groups)


def lin_reg_test():
    """

    :return:
    """
    features, group = log_reg_test_data()
    lg = LogisticRegression()
    lg(features, group)
    print(lg)
    # draw_logistic_data(features, group, lg.thetas)


def non_lin_reg_test():
    """

    :return:
    """
    features, group = log_reg_ellipsoid_test_data((0.08, -0.08, 0.6, 1.0, 1.0))
    lg = LogisticRegression()
    lg(features, group)
    print(lg)
    thetas = lg.thetas[1::] / -lg.thetas[0]
    print(lg.thetas/np.abs(lg.thetas[0]))


if __name__ == "__main__":
    lin_reg_test()
    non_lin_reg_test()
