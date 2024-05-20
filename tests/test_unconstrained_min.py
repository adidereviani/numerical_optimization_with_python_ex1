import unittest
from src.utils import *
from examples import *
from src.unconstrained_min import Minimization

class Minimization_tests(unittest.TestCase):

    def testQ1(self):
        test_q1 = Minimization()
        self.assertFalse(test_q1.minimizer_func(f=funcQ1, x0=np.array([1, 1], dtype=float), max_iter=funcQ1.max_iter, wolfe_cond_backtracking=False, title='Q1a')[2])
        self.assertFalse(test_q1.minimizer_func(f=funcQ1, x0=np.array([1, 1], dtype=float), max_iter=funcQ1.max_iter, wolfe_cond_backtracking=True, title='Q1b')[2])
        gradient_descent_dict = test_q1.function_dict
        self.assertTrue(test_q1.minimizer_func(f=funcQ1, x0=np.array([1, 1], dtype=float), max_iter=funcQ1.max_iter_hessian, hessian=True, wolfe_cond_backtracking=False, title='Q1c')[1] < 1e-5)
        self.assertTrue(test_q1.minimizer_func(f=funcQ1, x0=np.array([1, 1], dtype=float), max_iter=funcQ1.max_iter_hessian, hessian=True, wolfe_cond_backtracking=True, title='Q1d')[1] < 1e-5)
        newton_descent_dict = test_q1.function_dict
        plot_function_values(gradient_descent_dict, newton_descent_dict, title='Q1')
        plot_contour_2d(f=funcQ1, dict_a=gradient_descent_dict, dict_b=newton_descent_dict, title='Q1')

    def testQ2(self):
        test_q2 = Minimization()
        self.assertFalse(test_q2.minimizer_func(f=funcQ2, x0=np.array([1, 1], dtype=float), max_iter=funcQ2.max_iter, wolfe_cond_backtracking=False, title='Q2a')[2])
        self.assertFalse(test_q2.minimizer_func(f=funcQ2, x0=np.array([1, 1], dtype=float), max_iter=funcQ2.max_iter, wolfe_cond_backtracking=True, title='Q2b')[2])
        gradient_descent_dict = test_q2.function_dict
        self.assertTrue(test_q2.minimizer_func(f=funcQ2, x0=np.array([1, 1], dtype=float), max_iter=funcQ2.max_iter_hessian, hessian=True, wolfe_cond_backtracking=False, title='Q2c')[1] < 1e-5)
        self.assertTrue(test_q2.minimizer_func(f=funcQ2, x0=np.array([1, 1], dtype=float), max_iter=funcQ2.max_iter_hessian, hessian=True, wolfe_cond_backtracking=True, title='Q2d')[1] < 1e-5)
        newton_descent_dict = test_q2.function_dict
        plot_function_values(gradient_descent_dict, newton_descent_dict, title='Q2')
        plot_contour_2d(f=funcQ2, dict_a=gradient_descent_dict, dict_b=newton_descent_dict, title='Q2')

    def testQ3(self):
        test_q3 = Minimization()
        self.assertFalse(test_q3.minimizer_func(f=funcQ3, x0=np.array([1, 1], dtype=float), max_iter=funcQ3.max_iter, wolfe_cond_backtracking=False, title='Q3a')[2])
        self.assertFalse(test_q3.minimizer_func(f=funcQ3, x0=np.array([1, 1], dtype=float), max_iter=funcQ3.max_iter, wolfe_cond_backtracking=True, title='Q3b')[2])
        gradient_descent_dict = test_q3.function_dict
        self.assertTrue(test_q3.minimizer_func(f=funcQ3, x0=np.array([1, 1], dtype=float), max_iter=funcQ3.max_iter_hessian, hessian=True, wolfe_cond_backtracking=False, title='Q3c')[1] < 1e-5)
        self.assertTrue(test_q3.minimizer_func(f=funcQ3, x0=np.array([1, 1], dtype=float), max_iter=funcQ3.max_iter_hessian, hessian=True, wolfe_cond_backtracking=True, title='Q3d')[1] < 1e-5)
        newton_descent_dict = test_q3.function_dict
        plot_function_values(gradient_descent_dict, newton_descent_dict, title='Q3')
        plot_contour_2d(f=funcQ3, dict_a=gradient_descent_dict, dict_b=newton_descent_dict, title='Q3')

    def testRosenbrock(self):
        test_rosenbrock = Minimization()
        self.assertFalse(test_rosenbrock.minimizer_func(f=funcRosenbrock, x0=np.array([-1, 2], dtype=float), max_iter=funcRosenbrock.max_iter, wolfe_cond_backtracking=False, title='Rosenbrock_a')[2])
        self.assertTrue(test_rosenbrock.minimizer_func(f=funcRosenbrock, x0=np.array([-1, 2], dtype=float), max_iter=funcRosenbrock.max_iter, wolfe_cond_backtracking=True, title='Rosenbrock_b')[2])
        gradient_descent_dict = test_rosenbrock.function_dict
        self.assertTrue(test_rosenbrock.minimizer_func(f=funcRosenbrock, x0=np.array([-1, 2], dtype=float), max_iter=funcRosenbrock.max_iter_hessian, hessian=True, wolfe_cond_backtracking=False, title='Rosenbrock_c')[1] < 1e-5)
        self.assertTrue(test_rosenbrock.minimizer_func(f=funcRosenbrock, x0=np.array([-1, 2], dtype=float), max_iter=funcRosenbrock.max_iter_hessian, hessian=True, wolfe_cond_backtracking=True, title='Rosenbrock_d')[1] < 1e-5)
        newton_descent_dict = test_rosenbrock.function_dict
        plot_function_values(gradient_descent_dict, newton_descent_dict, title='Rosenbrock')
        plot_contour_2d(f=funcRosenbrock, dict_a=gradient_descent_dict, dict_b=newton_descent_dict, title='Rosenbrock')

    def testLinear(self):
        test_linear = Minimization()
        self.assertFalse(test_linear.minimizer_func(f=funcLinear, x0=np.array([1, 1], dtype=float), max_iter=funcLinear.max_iter, wolfe_cond_backtracking=False, title='Linear_a')[2])
        self.assertFalse(test_linear.minimizer_func(f=funcLinear, x0=np.array([1, 1], dtype=float), max_iter=funcLinear.max_iter, wolfe_cond_backtracking=True, title='Linear_b')[2])
        gradient_descent_dict = test_linear.function_dict
        self.assertFalse(test_linear.minimizer_func(f=funcLinear, x0=np.array([1, 1], dtype=float), max_iter=funcLinear.max_iter_hessian, hessian=True, wolfe_cond_backtracking=False, title='Linear_c')[2])
        self.assertFalse(test_linear.minimizer_func(f=funcLinear, x0=np.array([1, 1], dtype=float), max_iter=funcLinear.max_iter_hessian, hessian=True, wolfe_cond_backtracking=True, title='Linear_d')[2])
        newton_descent_dict = test_linear.function_dict
        plot_function_values(gradient_descent_dict, newton_descent_dict, title='Linear')
        plot_contour_2d(f=funcLinear, dict_a=gradient_descent_dict, dict_b=newton_descent_dict, title='Linear')

    def testExponential(self):
        test_exponential = Minimization()
        self.assertFalse(test_exponential.minimizer_func(f=funcExponential, x0=np.array([1, 1], dtype=float), max_iter=funcExponential.max_iter, wolfe_cond_backtracking=False, title='Exponential_a')[2])
        self.assertFalse(test_exponential.minimizer_func(f=funcExponential, x0=np.array([1, 1], dtype=float), max_iter=funcExponential.max_iter, wolfe_cond_backtracking=True, title='Exponential_b')[2])
        gradient_descent_dict = test_exponential.function_dict
        self.assertTrue(test_exponential.minimizer_func(f=funcExponential, x0=np.array([1, 1], dtype=float), max_iter=funcExponential.max_iter_hessian, hessian=True, wolfe_cond_backtracking=False, title='Exponential_c')[1] < 2.55927 + 1e-4)
        self.assertTrue(test_exponential.minimizer_func(f=funcExponential, x0=np.array([1, 1], dtype=float), max_iter=funcExponential.max_iter_hessian, hessian=True, wolfe_cond_backtracking=True, title='Exponential_d')[1] < 2.55927 + 1e-4)
        newton_descent_dict = test_exponential.function_dict
        plot_function_values(gradient_descent_dict, newton_descent_dict, title='Exponential')
        plot_contour_2d(f=funcExponential, dict_a=gradient_descent_dict, dict_b=newton_descent_dict, title='Exponential')

if __name__ == '__main__':
    unittest.main()