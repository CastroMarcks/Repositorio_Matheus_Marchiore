select
    a.jobid,
    a.subscriberkey,
    b.fromname as remetente,
    b.fromemail as email_remetente,
    b.emailname as nome_email
from
    ent.de_20240731_cdo_dashboard_funil_de_conversao as a
        join _job as b on a.jobid = b.jobid