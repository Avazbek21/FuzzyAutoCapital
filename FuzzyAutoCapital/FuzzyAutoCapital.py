import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
# skfuzzy - Этот пакет реализует множество полезных инструментов и функций для вычислений и проектов, 
# включающих нечеткую логику, также известную как серая логика.
# control - подпакет управления, предоставляющий API высокого уровня для проектирования нечетких систем.

from skfuzzy.control import ControlSystemSimulation

print("Задача: Определить насколько ""%"" рекомендуется выбранный город в роль столицы.")

# Категории(Критерии) выбора и их термы(3 - 5 шт.)
# 1. population  - составляет процент населения из 100% населения всей стране
# 2. territory   - составляет процент территории из 100% территории страны
# 3. Изначально расчитается самое максимальное расстояние(радиус) Rmax от границы до идеальной центральной точки 
#    DistanceFTB - составляет процент от Rmax, чтобы определить насколько близко расположен город к границе
# 4. celebrity   - Уровень знаменитости города по шкале [0; 100]
# Задание функций принадлежности для каждых терм

# Когда площадь города занимаем почти всю территорию государства, то однозначно выбирается этот город
territory = ctrl.Antecedent(np.arange(0, 100, 10), 'territory')
territory['low'] 	 	 = fuzz.trapmf(territory.universe, [0, 0, 5, 8])
territory['low_medium']  = fuzz.trapmf(territory.universe, [5, 10, 15, 20])
territory['medium']	 	 = fuzz.trapmf(territory.universe, [15, 21, 40, 45])
territory['medium_high'] = fuzz.trapmf(territory.universe, [40, 65, 75, 80])
territory['high'] 		 = fuzz.trapmf(territory.universe, [75, 80, 100, 100])

#Distance From The Border  - расстояние от границы до города
DistanceFTB = ctrl.Antecedent(np.arange(0, 100, 10), 'DistanceFTB')
DistanceFTB['low'] 	  = fuzz.trapmf(DistanceFTB.universe, [0, 1, 15, 20])
DistanceFTB['medium'] = fuzz.trapmf(DistanceFTB.universe, [15, 20, 50, 65]) #? последний параметр был 55
DistanceFTB['high']	  = fuzz.trapmf(DistanceFTB.universe, [60, 80, 100, 100]) #? первый параметр был 75

# Если остальные кретерии слабо проявляют себя, то тогда население города будет рещающим
population = ctrl.Antecedent(np.arange(0, 100, 10), 'population')
population['low'] 	 	  = fuzz.trapmf(population.universe, [0, 1, 15, 20])
population['medium'] 	  = fuzz.trapmf(population.universe, [15, 20, 50, 55])
population['medium_high'] = fuzz.trapmf(population.universe, [54, 65, 75, 80]) #? первый параметр был 55
population['high'] 		  = fuzz.trapmf(population.universe, [75, 80, 100, 100])

celebrity = ctrl.Antecedent(np.arange(0, 100, 10), 'celebrity')
celebrity['low'] 		 = fuzz.trapmf(celebrity.universe, [0, 0, 15, 20])
celebrity['medium']		 = fuzz.trapmf(celebrity.universe, [15, 20, 50, 55])
celebrity['medium_high'] = fuzz.trapmf(celebrity.universe, [54, 65, 75, 80]) #? первый параметр был 55
celebrity['high'] 		 = fuzz.trapmf(celebrity.universe, [75, 80, 100, 100])

# Визуализация нечётных множеств и правил

population.view()
territory.view()
DistanceFTB.view()
celebrity.view()


# Гауссовская функция (x - нечёткое множество, mean - среднее значение, sigma - отклонение от mean на *3)
# Определение нечёткой переменной вывода
PointsScored = ctrl.Consequent(np.arange(0, 101, 10), 'PointsScored') # [0; 100] с шагом 10
PointsScored['low'] 		= fuzz.gaussmf(PointsScored.universe, 0, 4.5) 			# Гауссовская функция с пиком в 0		[0;  12] +-(3*sigma)
PointsScored['low_medium']	= fuzz.trimf(PointsScored.universe, [10,  20, 35]) 		# Треугольная функция с пиком 25	[10;  35]
PointsScored['medium'] 	  	= fuzz.trapmf(PointsScored.universe, [30, 45, 60, 75])	# Трапециевидная функция с пиком в 45-60 [30; 75]
PointsScored['medium_high'] = fuzz.gaussmf(PointsScored.universe, 70,  8.33)		# Гауссовская функция с пиком в 70	[55; 95] +-(3*sigma)
PointsScored['high'] 		= fuzz.gaussmf(PointsScored.universe, 100, 8) 			# Гауссовская функция с пиком в 100 [;] +-(3*sigma)

print("Получение графиков")
print("Rules")
#------------------------------------------------------------------ готово
# territory['low']	|	territory['low_medium']	|	territory['medium']
R0 = ctrl.Rule(	(territory['low']	|	territory['low_medium']	|	territory['medium']) & \
				(DistanceFTB['low']	) & \
				(population['low']	) & \
				(celebrity['low']	|	celebrity['medium']	) \
				, PointsScored['low'])
#------------------------------------------------------------------ готово
R1 = ctrl.Rule(	(territory['low']	) & \
				(DistanceFTB['low']	) & \
				(population['low']	) & \
				(celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['low_medium'])
#------------------------------------------------------------------ готово
# territory['low']	|	territory['low_medium']	|	territory['medium']
R2 = ctrl.Rule(	(territory['low']	|	territory['low_medium']	|	territory['medium']) & \
				(DistanceFTB['low']		) & \
				(population['medium']	) & \
				(celebrity['low']		) \
				, PointsScored['low'])
#------------------------------------------------------------------ готово
# territory['low']	|	territory['low_medium']
R3 = ctrl.Rule(	(territory['low']	|	territory['low_medium']	) & \
				(DistanceFTB['low']		) & \
				(population['medium']	) & \
				(celebrity['medium']	|	celebrity['medium_high']	) \
				, PointsScored['low_medium'])
R4 = ctrl.Rule(	(territory['low']		|	territory['low_medium']	) & \
				(DistanceFTB['low']		) & \
				(population['medium']	) & \
				(celebrity['high']		) \
				, PointsScored['medium'])
#------------------------------------------------------------------ готово
R5 = ctrl.Rule(	(territory['low']			) & \
				(DistanceFTB['low']			) & \
				(population['medium_high']	) & \
				(celebrity['low']	|	celebrity['medium']	) \
				, PointsScored['low_medium'])
R6 = ctrl.Rule(	(territory['low']			) & \
				(DistanceFTB['low']			) & \
				(population['medium_high']	) & \
				(celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium'])
#------------------------------------------------------------------ готово
#territory['low']	|	territory['low_medium']
R7 = ctrl.Rule(	(territory['low']	|	territory['low_medium']	) & \
				(DistanceFTB['low']	) & \
				(population['high']	) & \
				(celebrity['low']	) \
				, PointsScored['low_medium'])
R8 = ctrl.Rule(	(territory['low']	|	territory['low_medium']	) & \
				(DistanceFTB['low']	) & \
				(population['high']	) & \
				(celebrity['medium']	|	celebrity['medium_high']	) \
				, PointsScored['medium'])
R9 = ctrl.Rule(	(territory['low']		|	territory['low_medium']	) & \
				(DistanceFTB['low']	) & \
				(population['high']	) & \
				(celebrity['high']	) \
				, PointsScored['medium_high'])
R10 = ctrl.Rule((territory['low']	|	territory['low_medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['low']	) & \
				(celebrity['low']	) \
				, PointsScored['low'])
#------------------------------------------------------------------ готово
R11 = ctrl.Rule((territory['low']	) & \
				(DistanceFTB['low']	) & \
				(population['low']	) & \
				(celebrity['medium']	) \
				, PointsScored['low_medium'])
R12 = ctrl.Rule((territory['low']	) & \
				(DistanceFTB['low']	) & \
				(population['low']	) & \
				(celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium'])
#------------------------------------------------------------------ готово
#territory['low']	|	territory['low_medium']
R13 = ctrl.Rule((territory['low']	|	territory['low_medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['medium']	) & \
				(celebrity['low']	) \
				, PointsScored['low_medium'])
#------------------------------------------------------------------ готово
R14 = ctrl.Rule((territory['low']		|	territory['low_medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['medium']	) & \
				(celebrity['medium']	|	celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium'])
R15 = ctrl.Rule((territory['low']		|	territory['low_medium']	) & \
				(DistanceFTB['medium']		) & \
				(population['medium_high']	) & \
				(celebrity['low']	|	celebrity['medium']	) \
				, PointsScored['medium'])
R16 = ctrl.Rule((territory['low']		|	territory['low_medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['medium_high']	) & \
				(celebrity['low']	|	celebrity['medium']		|	celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium_high'])
R17 = ctrl.Rule((territory['low']		|	territory['low_medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['high']	) & \
				(celebrity['low']	) \
				, PointsScored['medium'])
R18 = ctrl.Rule((territory['low']		|	territory['low_medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['high']	) & \
				(celebrity['medium']		|	celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium_high'])
R19 = ctrl.Rule((territory['low']		|	territory['low_medium']	) & \
				(DistanceFTB['high']	) & \
				(population['low']	) & \
				(celebrity['low']	) \
				, PointsScored['low'])
R20 = ctrl.Rule((territory['low']		|	territory['low_medium']	) & \
				(DistanceFTB['high']	) & \
				(population['low']	) & \
				(celebrity['medium']	) \
				, PointsScored['low_medium'])
#------------------------------------------------------------------ готово
# territory['low']	|	territory['medium']
R21 = ctrl.Rule((territory['low']	|	territory['medium']	) & \
				(DistanceFTB['high']		) & \
				(population['low']			) & \
				(celebrity['medium_high']	)\
				, PointsScored['medium']	)
R22 = ctrl.Rule((territory['low']		|	territory['medium']	) & \
				(DistanceFTB['high']	) & \
				(population['low']		) & \
				(celebrity['high']		) \
				, PointsScored['medium_high'])
#------------------------------------------------------------------ готово
R23 = ctrl.Rule((territory['low']		) & \
				(DistanceFTB['high']	) & \
				(population['medium']	) & \
				(celebrity['low']		) \
				, PointsScored['low_medium'])
R24 = ctrl.Rule((territory['low']		) & \
				(DistanceFTB['high']	) & \
				(population['medium']	) & \
				(celebrity['medium']	) \
				, PointsScored['medium'])
#------------------------------------------------------------------ готово
# territory['low']	|	territory['low_medium']
R25 = ctrl.Rule((territory['low']	|	territory['low_medium']	) & \
				(DistanceFTB['high']	) & \
				(population['medium']	) & \
				(celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium_high'])
#------------------------------------------------------------------	готово
# territory['low']	|	territory['low_medium']	|	territory['medium'] 
R26 = ctrl.Rule((territory['low']	|	territory['low_medium']	|	territory['medium']	) & \
				(DistanceFTB['high']	) & \
				(population['medium_high']	) & \
				(celebrity['low']	|	celebrity['medium']	) \
				, PointsScored['medium_high'])
R27 = ctrl.Rule((territory['low']	|	territory['low_medium']	|	territory['medium']	) & \
				(DistanceFTB['high']	) & \
				(population['medium_high']	) & \
				(celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['high'])
#------------------------------------------------------------------ готово
R28 = ctrl.Rule((territory['low']		) & \
				(DistanceFTB['high']	) & \
				(population['high']		) & \
				(celebrity['low']		) \
				, PointsScored['medium_high'])
R29 = ctrl.Rule((territory['low']	|	territory['low_medium']	) & \
				(DistanceFTB['high']	) & \
				(population['high']		) & \
				(celebrity['medium']		|	celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['high'])
#------------------------------------------------------------------ готово
# territory['low_medium']
R30 = ctrl.Rule((territory['low_medium']	) & \
				(DistanceFTB['low']			) & \
				(population['low']			) & \
				(celebrity['medium_high']	) \
				, PointsScored['low_medium'])
R31 = ctrl.Rule((territory['low_medium']	) & \
				(DistanceFTB['low']			) & \
				(population['low']			) & \
				(celebrity['high']			) \
				, PointsScored['medium'])
R32 = ctrl.Rule((territory['low_medium']	) & \
				(DistanceFTB['low']	) & \
				(population['medium_high']	) & \
				(celebrity['low']	) \
				, PointsScored['low_medium'])
R33 = ctrl.Rule((territory['low_medium']	) & \
				(DistanceFTB['low']	) & \
				(population['medium_high']	) & \
				(celebrity['medium']	|	celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium'])
R34 = ctrl.Rule((territory['low_medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['low']	) & \
				(celebrity['medium']	|	celebrity['medium_high']) \
				, PointsScored['low_medium'])
#------------------------------------------------------------------ готово
# territory['low_medium']	|	territory['medium']
R35 = ctrl.Rule((territory['low_medium']	|	territory['medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['low']		) & \
				(celebrity['high']		) \
				, PointsScored['medium'])
#------------------------------------------------------------------ готово
# territory['low_medium']
R36 = ctrl.Rule((territory['low_medium']	) & \
				(DistanceFTB['medium']		) & \
				(population['medium']		) & \
				(celebrity['medium']	|	celebrity['medium_high']	) \
				, PointsScored['medium'])
R37 = ctrl.Rule((territory['low_medium']	) & \
				(DistanceFTB['medium']		) & \
				(population['medium']		) & \
				(celebrity['high']			) \
				, PointsScored['medium_high'])
R38 = ctrl.Rule((territory['low']		|	territory['low_medium']	) & \
				(DistanceFTB['medium']		) & \
				(population['medium_high']	) & \
				(celebrity['low']			) \
				, PointsScored['medium'])
R39 = ctrl.Rule((territory['low']		|	territory['low_medium']	) & \
				(DistanceFTB['medium']		) & \
				(population['medium_high']	) & \
				(celebrity['medium']		|	celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium_high'])
R40 = ctrl.Rule((territory['low']		|	territory['low_medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['low']	|	population['medium']	|	population['medium_high']	|	population['high']	) & \
				(celebrity['low']	|	celebrity['medium']		|	celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium_high'])
#------------------------------------------------------------------ готово
# territory['low_medium']	|	territory['medium']
R41 = ctrl.Rule((territory['low_medium']	|	territory['medium']	) & \
				(DistanceFTB['high']	) & \
				(population['low']	) & \
				(celebrity['low']	|	celebrity['medium']	) \
				, PointsScored['low_medium'])
#------------------------------------------------------------------ готово
# territory['low_medium']
R42 = ctrl.Rule((territory['low_medium']	) & \
				(DistanceFTB['high']	) & \
				(population['low']	) & \
				(celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium'])
R43 = ctrl.Rule((territory['low_medium']	) & \
				(DistanceFTB['high']		) & \
				(population['medium']		) & \
				(celebrity['low']	|	celebrity['medium']	) \
				, PointsScored['medium'])
R44 = ctrl.Rule((territory['low_medium']	) & \
				(DistanceFTB['high']		) & \
				(population['high']			) & \
				(celebrity['low']	|	celebrity['medium']	) \
				, PointsScored['medium_high'])
R45 = ctrl.Rule((territory['low_medium']	) & \
				(DistanceFTB['high']		) & \
				(population['high']			) & \
				(celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['high'])
#------------------------------------------------------------------ готово
# territory['medium']
R46 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['low']	) & \
				(population['medium']	) & \
				(celebrity['medium']	) \
				, PointsScored['low_medium'])
R47 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['low']	) & \
				(population['medium']	) & \
				(celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium'])
R48 = ctrl.Rule((territory['medium']		) & \
				(DistanceFTB['low']			) & \
				(population['medium_high']	) & \
				(celebrity['low']	|	celebrity['medium']	) \
				, PointsScored['medium'])
R49 = ctrl.Rule((territory['medium']		) & \
				(DistanceFTB['low']			) & \
				(population['medium_high']	) & \
				(celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium_high'])
R50 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['low']		) & \
				(population['high']		) & \
				(celebrity['low']	|	celebrity['medium']	) \
				, PointsScored['medium'])
R51 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['low']		) & \
				(population['high']		) & \
				(celebrity['medium_high']) \
				, PointsScored['medium_high'])
R52 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['low']		) & \
				(population['high']		) & \
				(celebrity['high']		) \
				, PointsScored['high'])
R53 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['low']	) & \
				(celebrity['low']	|	celebrity['medium']		|	celebrity['medium_high']	) \
				, PointsScored['low_medium'])
R54 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['medium']	) & \
				(celebrity['low']	|	celebrity['medium']		|	celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium'])
R55 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['medium_high']	) & \
				(celebrity['low']	|	celebrity['medium']	) \
				, PointsScored['medium_high'])
R56 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['medium_high']	) & \
				(celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['high'])
R57 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['high']		) & \
				(celebrity['low']	|	celebrity['medium']	) \
				, PointsScored['medium_high'])
R58 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['medium']	) & \
				(population['high']		) & \
				(celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['high'])
R59 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['high']	) & \
				(population['medium']	) & \
				(celebrity['low']		) \
				, PointsScored['medium'])
R60 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['high']	) & \
				(population['medium']	) & \
				(celebrity['medium']) \
				, PointsScored['medium_high'])
R61 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['high']	) & \
				(population['medium']	) & \
				(celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['high'])
R62 = ctrl.Rule((territory['medium']	) & \
				(DistanceFTB['high']	) & \
				(population['high']	) & \
				(celebrity['low']	|	celebrity['medium']		|	celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['high'])
#------------------------------------------------------------------ готово
#(territory['medium_high']
R63 = ctrl.Rule((territory['medium_high']	) & \
				(DistanceFTB['low']	) & \
				(population['low']	|	population['medium']	|	population['medium_high']	|	population['high']	) & \
				(celebrity['low']	|	celebrity['medium']		|	celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['high'])
R64 = ctrl.Rule((territory['medium_high']	) & \
				(DistanceFTB['medium']	|	DistanceFTB['high']	) & \
				(population['low']	|	population['medium']	|	population['medium_high']	|	population['high']	) & \
				(celebrity['low']	|	celebrity['medium']		|	celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium_high'])
#------------------------------------------------------------------ готово
#(territory['high']
R65 = ctrl.Rule((territory['high']	) & \
				(DistanceFTB['low']	|	DistanceFTB['medium']	) & \
				(population['low']	|	population['medium']	|	population['medium_high']	|	population['high']	) & \
				(celebrity['low']	|	celebrity['medium']		|	celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['high'])
R66 = ctrl.Rule((territory['high']	) & \
				(DistanceFTB['high']	) & \
				(population['low']	|	population['medium']	|	population['medium_high']	|	population['high']	) & \
				(celebrity['low']	|	celebrity['medium']		|	celebrity['medium_high']	|	celebrity['high']	) \
				, PointsScored['medium_high'])

Rules=[R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31, R32, R33, R34, R35, R36, R37, R38, R39, R40, R41, R42, R43, R44, R45, R46, R47, R48, R49, R50, R51, R52, R53, R54, R55, R56, R57, R58, R59, R60, R61, R62, R63, R64, R65, R66]
# Создание системы управления
PointsScored_ctrl1 = ctrl.ControlSystem(Rules)
PointsScored_ctrl2 = ctrl.ControlSystem(Rules)
PointsScored_ctrl3 = ctrl.ControlSystem(Rules)
PointsScored_ctrl4 = ctrl.ControlSystem(Rules)
PointsScored_ctrl5 = ctrl.ControlSystem(Rules)
#PointsScored_simulation = ctrl.ControlSystemSimulation(PointsScored_ctrl)

#объявление "кандидатов" в роль столицы государства
Town1 = ctrl.ControlSystemSimulation(PointsScored_ctrl1)
Town2 = ctrl.ControlSystemSimulation(PointsScored_ctrl2)
Town3 = ctrl.ControlSystemSimulation(PointsScored_ctrl3)
Town4 = ctrl.ControlSystemSimulation(PointsScored_ctrl4)
Town5 = ctrl.ControlSystemSimulation(PointsScored_ctrl5)
# Ввод данных о городе №1 в кандидаты
Town1.input['population']	= 99
Town1.input['territory']	= 25
Town1.input['DistanceFTB']	= 33
Town1.input['celebrity'] 	= 40
Town1.compute()	# Вычисление рекомендации
# Ввод данных о городе №2 в кандидаты
Town2.input['population']	= 50
Town2.input['territory']	= 33
Town2.input['DistanceFTB']	= 49
Town2.input['celebrity'] 	= 19
Town2.compute()	# Вычисление рекомендации
# Ввод данных о городе №3 в кандидаты
Town3.input['population']	= 75
Town3.input['territory']	= 88
Town3.input['DistanceFTB']	= 7
Town3.input['celebrity'] 	= 64
Town3.compute()	# Вычисление рекомендации
# Ввод данных о городе №4 в кандидаты
Town4.input['population']	= 30
Town4.input['territory']	= 10
Town4.input['DistanceFTB']	= 60
Town4.input['celebrity'] 	= 76
Town4.compute()	# Вычисление рекомендации
# Ввод данных о городе №5 в кандидаты
Town5.input['population']	= 38
Town5.input['territory']	= 4
Town5.input['DistanceFTB']	= 20
Town5.input['celebrity'] 	= 95
Town5.compute()	# Вычисление рекомендации

print("Степень достоверности городов в роле столицы государства (в %):")
print("Город №1 ""Авалонне""     = ", Town1.output['PointsScored'])
print("Город №2 ""Альквалондэ""  = ", Town2.output['PointsScored'])
print("Город №3 ""Барад Эйтель"" = ", Town3.output['PointsScored'])
print("Город №4 ""Форменос""     = ", Town4.output['PointsScored'])
print("Город №5 ""Агларонд""     = ", Town5.output['PointsScored'])
PointsScored.view()
