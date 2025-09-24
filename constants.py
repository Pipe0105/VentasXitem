# constants.py

SEDE_NAME_MAP = {
    "mercamio": {"1": "Calle 5ta","2": "La 39","3": "Plaza","4": "Jardín","5": "C. Sur","6": "Palmira"},
    "mtodo": {"1": "Floresta","2": "Floralia","3": "Guadua"},
    "bogota": {"1": "Calle 80","2": "Chía"},
}

MONTH_ABBR_ES = {
    1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"
}
MONTH_FULL_ES = {
    1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
}

ALIASES = {
    "empresa": {"empresa","compania","compañia","company"},
    "fecha_dcto": {"fecha_dcto","fecha","fecha_doc","fecha documento","fecha_documento"},
    "id_co": {"id_co","sede","tienda","local","centro"},
    "id_item": {"id_item","item","codigo_item","sku","cod_item"},
    "und_dia": {"und_dia","und","unid","unidades","cantidad","cant"},
    "descripcion": {"descripcion","descripción","desc","detalle"},
    "ub_unit": {"ub_unit","unidad","unidad_medida","um","u.m.","u_m"},
    "ub_factor": {"ub_factor","factor","contenido","presentacion","presentación","unid_x","unidx","und_pack","unidades_por","pack","x"},
}
