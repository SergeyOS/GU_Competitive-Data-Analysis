# GU_Competitive-Data-Analysis

Описание:
1. <a href='https://github.com/SergeyOS/GU_Competitive-Data-Analysis/blob/main.py'> main.py </a>. Содержит:
* параметры запуска
* основной pipeline
* функцию запуска
2. <a href='https://github.com/SergeyOS/GU_Competitive-Data-Analysis/blob/main/FeatureEngineering/client_profile.py'>FeatureEngineering/client_profile.py</a>. создает признаки на основе профиля клиентов. 
3. <a href='https://github.com/SergeyOS/GU_Competitive-Data-Analysis/blob/main/FeatureEngineering/app_history.py'>FeatureEngineering/app_history.py</a>. создает признаки на основе истории ранее поданных заявок. 
4. <a href='https://github.com/SergeyOS/GU_Competitive-Data-Analysis/blob/main/FeatureEngineering/payment_stats.py'>FeatureEngineering/payment_stats.py</a>. создает признаки на основе истории платежей. 
5. <a href='https://github.com/SergeyOS/GU_Competitive-Data-Analysis/blob/main/FeatureEngineering/features_transform.py'>FeatureEngineering/features_transform.py</a>. содержит набор классов для дополнительной обработки признаков:
  * FeaturesTransform - основной класс изменяющий набор признаков, в том числе добавляющий PCA  признаки на базе числовых признаков
  * DFFeatureUnion - объединяет признаки
  * DFStandardScaler - оболочка для применения StandardScaler (не применялся в итоговой модели)
  * PCATransformer - для создания PCA признаков
  * MulticollinearityThreshold - оболочка для удаления мультиколлинеарных признаков (не применялся в итоговой модели)
  * GeneratorNumFeatures - генерирует на базе числовых признаков новые с использованием математических операций
6. <a href='https://github.com/SergeyOS/GU_Competitive-Data-Analysis/blob/main/FeatureEngineering/features_selection.py'>features_selection.py</a> Осуществляет отбор признаков:
  * permutation importance (не применялся в итоговой модели)
  * shap_importance
7. <a href='https://github.com/SergeyOS/GU_Competitive-Data-Analysis/blob/main/FeatureEngineering/warp_catboost.py'>warp_catboost.py</a> Оболочка для кросвалидации основной модели. 
  
