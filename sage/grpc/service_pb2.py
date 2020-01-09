# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: service.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='service.proto',
  package='sage',
  syntax='proto3',
  serialized_options=_b('\n\026fr.univnantes.gdd.sageB\nSageSPARQLP\001\242\002\003HLW'),
  serialized_pb=_b('\n\rservice.proto\x12\x04sage\"H\n\tSageQuery\x12\r\n\x05query\x18\x01 \x01(\t\x12\x19\n\x11\x64\x65\x66\x61ult_graph_uri\x18\x02 \x01(\t\x12\x11\n\tnext_link\x18\x03 \x01(\t\"*\n\x07\x42inding\x12\x10\n\x08variable\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\"+\n\nBindingSet\x12\x1d\n\x06values\x18\x01 \x03(\x0b\x32\r.sage.Binding\"V\n\x0cSageResponse\x12\"\n\x08\x62indings\x18\x01 \x03(\x0b\x32\x10.sage.BindingSet\x12\x0f\n\x07is_done\x18\x02 \x01(\x08\x12\x11\n\tnext_link\x18\x03 \x01(\t2<\n\nSageSPARQL\x12.\n\x05Query\x12\x0f.sage.SageQuery\x1a\x12.sage.SageResponse\"\x00\x42,\n\x16\x66r.univnantes.gdd.sageB\nSageSPARQLP\x01\xa2\x02\x03HLWb\x06proto3')
)




_SAGEQUERY = _descriptor.Descriptor(
  name='SageQuery',
  full_name='sage.SageQuery',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='query', full_name='sage.SageQuery.query', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='default_graph_uri', full_name='sage.SageQuery.default_graph_uri', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='next_link', full_name='sage.SageQuery.next_link', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=23,
  serialized_end=95,
)


_BINDING = _descriptor.Descriptor(
  name='Binding',
  full_name='sage.Binding',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='variable', full_name='sage.Binding.variable', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='sage.Binding.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=97,
  serialized_end=139,
)


_BINDINGSET = _descriptor.Descriptor(
  name='BindingSet',
  full_name='sage.BindingSet',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='values', full_name='sage.BindingSet.values', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=141,
  serialized_end=184,
)


_SAGERESPONSE = _descriptor.Descriptor(
  name='SageResponse',
  full_name='sage.SageResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='bindings', full_name='sage.SageResponse.bindings', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='is_done', full_name='sage.SageResponse.is_done', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='next_link', full_name='sage.SageResponse.next_link', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=186,
  serialized_end=272,
)

_BINDINGSET.fields_by_name['values'].message_type = _BINDING
_SAGERESPONSE.fields_by_name['bindings'].message_type = _BINDINGSET
DESCRIPTOR.message_types_by_name['SageQuery'] = _SAGEQUERY
DESCRIPTOR.message_types_by_name['Binding'] = _BINDING
DESCRIPTOR.message_types_by_name['BindingSet'] = _BINDINGSET
DESCRIPTOR.message_types_by_name['SageResponse'] = _SAGERESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SageQuery = _reflection.GeneratedProtocolMessageType('SageQuery', (_message.Message,), {
  'DESCRIPTOR' : _SAGEQUERY,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:sage.SageQuery)
  })
_sym_db.RegisterMessage(SageQuery)

Binding = _reflection.GeneratedProtocolMessageType('Binding', (_message.Message,), {
  'DESCRIPTOR' : _BINDING,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:sage.Binding)
  })
_sym_db.RegisterMessage(Binding)

BindingSet = _reflection.GeneratedProtocolMessageType('BindingSet', (_message.Message,), {
  'DESCRIPTOR' : _BINDINGSET,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:sage.BindingSet)
  })
_sym_db.RegisterMessage(BindingSet)

SageResponse = _reflection.GeneratedProtocolMessageType('SageResponse', (_message.Message,), {
  'DESCRIPTOR' : _SAGERESPONSE,
  '__module__' : 'service_pb2'
  # @@protoc_insertion_point(class_scope:sage.SageResponse)
  })
_sym_db.RegisterMessage(SageResponse)


DESCRIPTOR._options = None

_SAGESPARQL = _descriptor.ServiceDescriptor(
  name='SageSPARQL',
  full_name='sage.SageSPARQL',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=274,
  serialized_end=334,
  methods=[
  _descriptor.MethodDescriptor(
    name='Query',
    full_name='sage.SageSPARQL.Query',
    index=0,
    containing_service=None,
    input_type=_SAGEQUERY,
    output_type=_SAGERESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_SAGESPARQL)

DESCRIPTOR.services_by_name['SageSPARQL'] = _SAGESPARQL

# @@protoc_insertion_point(module_scope)
