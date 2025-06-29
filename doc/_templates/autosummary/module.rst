{{ name | escape | underline}}

.. automodule:: {{ fullname }}
    :members:
    :private-members:

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
