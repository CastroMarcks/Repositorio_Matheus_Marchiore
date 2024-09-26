select
    a.jobid,
    a.subscriberkey,
    min(b.eventdate) as data_abertura,
    format(min(b.eventdate),'yyyy-MM-dd') as data_abertura_format
from
    ent.de_20240731_cdo_dashboard_funil_de_conversao as a
        join _open as b on a.jobid = b.jobid and a.subscriberkey = b.subscriberkey
group by
    a.jobid,
    a.subscriberkey