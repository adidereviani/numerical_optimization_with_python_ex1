from tests.examples import *


class Minimization:

    def __init__(self):
        self.function_dict = {}

    def minimizer_func(self, f, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=None, hessian=False, wolfe_cond_backtracking=False, title=None):
        max_condition = 1e+5
        f_prev = []
        x_prev = []
        f.hessian = hessian
        status = False
        if title == 'Q2b':
            step_length = 1e-1
        elif title == 'Linear_b':
            step_length = 0.15
        else:
            step_length = 1e-2
        if hessian:
            func_name = 'Newton Descent'
            x = x0.copy()
            f_x, g_x, h_x = f.evaluate_function(x)
            x_prev.append(x)
            f_prev.append(f_x)
            print("Iteration number 0:\nFunction location is: {}\nFunction value is: {}".format(x.T, f_x))
            print('-' * 100)
            for i in range(1, max_iter+1):
                condition = np.linalg.cond(h_x)
                if condition > max_condition:
                    print("\nInversion condition number exceeds maximum condition number")
                    status = False
                    break
                if wolfe_cond_backtracking:
                    alpha = Minimization.wolfe_cond_backtracking(f=f, x=x, hessian=hessian)
                else:
                    alpha = 1
                direction = alpha * np.linalg.solve(h_x, -g_x)
                x = x.copy()
                x += direction
                f_x, g_x, h_x = f.evaluate_function(x)
                x_prev.append(x)
                f_prev.append(f_x)
                if len(f_prev) and len(x_prev) > 1:
                    function_values_diff = f_prev[-2] - f_prev[-1]
                    function_location_diff = np.linalg.norm(x_prev[-2] - x_prev[-1])
                    if abs(function_values_diff) < obj_tol or function_location_diff < param_tol:
                        print('\nProcess status: Successfully reached the numerical tolerance for termination')
                        x_prev.pop()
                        f_prev.pop()
                        status = True
                        break
                if abs(f_x) >= 1e+6:
                    print('\nThe functions value exceeds acceptable limits')
                    x_prev.pop()
                    f_prev.pop()
                    status = False
                    break
                print("Iteration number {}:\nFunction location is: {}\nFunction value is: {}".format(i, x.T, f_x))
                print('-' * 100)

        else:
            func_name = 'Gradient Descent'
            x = x0.copy()
            f_x, g_x = f.evaluate_function(x)
            x_prev.append(x)
            f_prev.append(f_x)
            print("Iteration number 0:\nFunction location is: {}\nFunction value is: {}".format(x.T, f_x))
            print('-' * 100)
            for i in range(1, max_iter+1):
                if wolfe_cond_backtracking:
                    alpha = Minimization.wolfe_cond_backtracking(f=f, x=x, step_size=step_length, hessian=hessian)
                else:
                    alpha = 1
                x = x.copy()
                x -= alpha*step_length*g_x
                f_x, g_x = f.evaluate_function(x)
                x_prev.append(x)
                f_prev.append(f_x)
                if abs(f_x) >= 1e+4:
                    print('\nThe functions value exceeds acceptable limits')
                    x_prev.pop()
                    f_prev.pop()
                    status = False
                    break
                if len(x_prev) and len(x_prev) > 1:
                    function_values_diff = f_prev[-2]-f_prev[-1]
                    function_location_diff = np.linalg.norm(x_prev[-2]-x_prev[-1])
                    if abs(function_values_diff) < obj_tol or function_location_diff < param_tol:
                        print('\nProcess status: Successfully reached the numerical tolerance for termination')
                        x_prev.pop()
                        f_prev.pop()
                        status = True
                        break
                print("Iteration number {}:\nFunction location is: {}\nFunction value is: {}".format(i, x.T, f_x))
                print('-' * 100)
        self.function_dict = {'Minimization class': func_name, 'Function location': np.vstack(x_prev).T, 'Function value': f_prev}
        print("\nFinal {} function location is {}\nFinal {} function value is {}\nStatus is {}\n".format(title, x_prev[-1].T, title, f_prev[-1], status))
        return x_prev[-1], f_prev[-1], status

    @staticmethod
    def wolfe_cond_backtracking(f, x, step_size=None, alpha=1, wcc=0.01, bc=0.5, hessian=False):
        f_x_1, g_x_1, *h_x_1 = f.evaluate_function(x)
        if hessian:
            direction = np.linalg.solve(h_x_1[0], -g_x_1)
            condition = lambda a: f.evaluate_function(x + a * direction)[0] > f_x_1 + wcc * a * g_x_1.T.dot(direction)
        else:
            direction = -step_size * g_x_1
            condition = lambda a: f.evaluate_function(x + a * direction)[0] > f_x_1 + wcc * a * g_x_1.T.dot(direction)

        while condition(alpha):
            alpha *= bc

        return alpha
