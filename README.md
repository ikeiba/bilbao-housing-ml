# bilbao-housing-ml
Machine Learning project on Bilbao housing


# DATA CLEANING:

#   Usable square meters: agrupar por zone o neighbourhood, y mirar el porcentaje de cambio entre usable y constructed y aplicarselo a las que no tienen (PROBAR LINEAR REGRESSION) --> Enetz

#   Exterior: las que son nuevas son exterior, el resto CLASSIFICATION --> Gonzalo

#   Floor: casas o bajos o entreplantas (MIRARLO POR ENCIMA, bajo 0, casa -1) --> Xabi

#   Condition: relacionada con la zona y el precio (CLASSIFICATION) --> David
#   HECHO: no ha hecho falta hacer classification, ya que todas las casas con valor nulo en esta columna era porque eran #   de obra nueva, asique he sustiuido NA por "Nuevo".

#   Heating: CLASSIFICATION --> Enetz

#   Year: en los que es obra nueva poner 2025, y en los que no poner la MEDIANA de la zona (o hacer LINEAR REGRESSION) --> David

#   Consumption/emision values: mirar las que tienen etiquetas y ponerles la mitad del rango --> Xabi

#   Consumption/emission labels: VAMOS A USAR PARA HACER CLASSIFICATION --> Iker


# CLASIFICATION:

#   



