select
    jobid,
    SubscriberKey,
    min(EventDate) as data_envio,
    format(min(EventDate),'yyyy-MM-dd') as data_envio_format
from
    _Sent
group by
    jobid,
    SubscriberKey