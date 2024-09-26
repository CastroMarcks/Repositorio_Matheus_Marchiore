select
    a.jobid,
    a.subscriberkey,
    b.created_at as data_ini_cad_cdo,
    format(b.created_at,'yyyy-MM-dd') as data_ini_cad_cdo_format,
    b.updated_at as data_updt_cad_cdo,
    format(b.updated_at,'yyyy-MM-dd') as data_updt_cad_cdo_format
from
    ent.de_20240731_cdo_dashboard_funil_de_conversao as a
        join ent.leadsv2_attributes as b on a.subscriberkey = b.lead_id
where
    b.attribute_name = 'onboarding_step'
    and b.value is not null
    and b.product = 'conta digital'