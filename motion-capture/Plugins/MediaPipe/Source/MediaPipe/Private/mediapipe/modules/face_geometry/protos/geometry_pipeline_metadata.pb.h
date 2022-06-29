// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mediapipe/modules/face_geometry/protos/geometry_pipeline_metadata.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fgeometry_5fpipeline_5fmetadata_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fgeometry_5fpipeline_5fmetadata_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3011000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3011004 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/unknown_field_set.h>
#include "mediapipe/modules/face_geometry/protos/mesh_3d.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fgeometry_5fpipeline_5fmetadata_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fgeometry_5fpipeline_5fmetadata_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fgeometry_5fpipeline_5fmetadata_2eproto;
namespace mediapipe {
namespace face_geometry {
class GeometryPipelineMetadata;
class GeometryPipelineMetadataDefaultTypeInternal;
extern GeometryPipelineMetadataDefaultTypeInternal _GeometryPipelineMetadata_default_instance_;
class WeightedLandmarkRef;
class WeightedLandmarkRefDefaultTypeInternal;
extern WeightedLandmarkRefDefaultTypeInternal _WeightedLandmarkRef_default_instance_;
}  // namespace face_geometry
}  // namespace mediapipe
PROTOBUF_NAMESPACE_OPEN
template<> ::mediapipe::face_geometry::GeometryPipelineMetadata* Arena::CreateMaybeMessage<::mediapipe::face_geometry::GeometryPipelineMetadata>(Arena*);
template<> ::mediapipe::face_geometry::WeightedLandmarkRef* Arena::CreateMaybeMessage<::mediapipe::face_geometry::WeightedLandmarkRef>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace mediapipe {
namespace face_geometry {

enum InputSource : int {
  DEFAULT = 0,
  FACE_LANDMARK_PIPELINE = 1,
  FACE_DETECTION_PIPELINE = 2
};
bool InputSource_IsValid(int value);
constexpr InputSource InputSource_MIN = DEFAULT;
constexpr InputSource InputSource_MAX = FACE_DETECTION_PIPELINE;
constexpr int InputSource_ARRAYSIZE = InputSource_MAX + 1;

const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* InputSource_descriptor();
template<typename T>
inline const std::string& InputSource_Name(T enum_t_value) {
  static_assert(::std::is_same<T, InputSource>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function InputSource_Name.");
  return ::PROTOBUF_NAMESPACE_ID::internal::NameOfEnum(
    InputSource_descriptor(), enum_t_value);
}
inline bool InputSource_Parse(
    const std::string& name, InputSource* value) {
  return ::PROTOBUF_NAMESPACE_ID::internal::ParseNamedEnum<InputSource>(
    InputSource_descriptor(), name, value);
}
// ===================================================================

class WeightedLandmarkRef :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.face_geometry.WeightedLandmarkRef) */ {
 public:
  WeightedLandmarkRef();
  virtual ~WeightedLandmarkRef();

  WeightedLandmarkRef(const WeightedLandmarkRef& from);
  WeightedLandmarkRef(WeightedLandmarkRef&& from) noexcept
    : WeightedLandmarkRef() {
    *this = ::std::move(from);
  }

  inline WeightedLandmarkRef& operator=(const WeightedLandmarkRef& from) {
    CopyFrom(from);
    return *this;
  }
  inline WeightedLandmarkRef& operator=(WeightedLandmarkRef&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const WeightedLandmarkRef& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const WeightedLandmarkRef* internal_default_instance() {
    return reinterpret_cast<const WeightedLandmarkRef*>(
               &_WeightedLandmarkRef_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(WeightedLandmarkRef& a, WeightedLandmarkRef& b) {
    a.Swap(&b);
  }
  inline void Swap(WeightedLandmarkRef* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline WeightedLandmarkRef* New() const final {
    return CreateMaybeMessage<WeightedLandmarkRef>(nullptr);
  }

  WeightedLandmarkRef* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<WeightedLandmarkRef>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const WeightedLandmarkRef& from);
  void MergeFrom(const WeightedLandmarkRef& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(WeightedLandmarkRef* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.face_geometry.WeightedLandmarkRef";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fgeometry_5fpipeline_5fmetadata_2eproto);
    return ::descriptor_table_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fgeometry_5fpipeline_5fmetadata_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kLandmarkIdFieldNumber = 1,
    kWeightFieldNumber = 2,
  };
  // optional uint32 landmark_id = 1;
  bool has_landmark_id() const;
  private:
  bool _internal_has_landmark_id() const;
  public:
  void clear_landmark_id();
  ::PROTOBUF_NAMESPACE_ID::uint32 landmark_id() const;
  void set_landmark_id(::PROTOBUF_NAMESPACE_ID::uint32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::uint32 _internal_landmark_id() const;
  void _internal_set_landmark_id(::PROTOBUF_NAMESPACE_ID::uint32 value);
  public:

  // optional float weight = 2;
  bool has_weight() const;
  private:
  bool _internal_has_weight() const;
  public:
  void clear_weight();
  float weight() const;
  void set_weight(float value);
  private:
  float _internal_weight() const;
  void _internal_set_weight(float value);
  public:

  // @@protoc_insertion_point(class_scope:mediapipe.face_geometry.WeightedLandmarkRef)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::uint32 landmark_id_;
  float weight_;
  friend struct ::TableStruct_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fgeometry_5fpipeline_5fmetadata_2eproto;
};
// -------------------------------------------------------------------

class GeometryPipelineMetadata :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:mediapipe.face_geometry.GeometryPipelineMetadata) */ {
 public:
  GeometryPipelineMetadata();
  virtual ~GeometryPipelineMetadata();

  GeometryPipelineMetadata(const GeometryPipelineMetadata& from);
  GeometryPipelineMetadata(GeometryPipelineMetadata&& from) noexcept
    : GeometryPipelineMetadata() {
    *this = ::std::move(from);
  }

  inline GeometryPipelineMetadata& operator=(const GeometryPipelineMetadata& from) {
    CopyFrom(from);
    return *this;
  }
  inline GeometryPipelineMetadata& operator=(GeometryPipelineMetadata&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const GeometryPipelineMetadata& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const GeometryPipelineMetadata* internal_default_instance() {
    return reinterpret_cast<const GeometryPipelineMetadata*>(
               &_GeometryPipelineMetadata_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(GeometryPipelineMetadata& a, GeometryPipelineMetadata& b) {
    a.Swap(&b);
  }
  inline void Swap(GeometryPipelineMetadata* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline GeometryPipelineMetadata* New() const final {
    return CreateMaybeMessage<GeometryPipelineMetadata>(nullptr);
  }

  GeometryPipelineMetadata* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<GeometryPipelineMetadata>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const GeometryPipelineMetadata& from);
  void MergeFrom(const GeometryPipelineMetadata& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(GeometryPipelineMetadata* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "mediapipe.face_geometry.GeometryPipelineMetadata";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fgeometry_5fpipeline_5fmetadata_2eproto);
    return ::descriptor_table_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fgeometry_5fpipeline_5fmetadata_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kProcrustesLandmarkBasisFieldNumber = 2,
    kCanonicalMeshFieldNumber = 1,
    kInputSourceFieldNumber = 3,
  };
  // repeated .mediapipe.face_geometry.WeightedLandmarkRef procrustes_landmark_basis = 2;
  int procrustes_landmark_basis_size() const;
  private:
  int _internal_procrustes_landmark_basis_size() const;
  public:
  void clear_procrustes_landmark_basis();
  ::mediapipe::face_geometry::WeightedLandmarkRef* mutable_procrustes_landmark_basis(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::face_geometry::WeightedLandmarkRef >*
      mutable_procrustes_landmark_basis();
  private:
  const ::mediapipe::face_geometry::WeightedLandmarkRef& _internal_procrustes_landmark_basis(int index) const;
  ::mediapipe::face_geometry::WeightedLandmarkRef* _internal_add_procrustes_landmark_basis();
  public:
  const ::mediapipe::face_geometry::WeightedLandmarkRef& procrustes_landmark_basis(int index) const;
  ::mediapipe::face_geometry::WeightedLandmarkRef* add_procrustes_landmark_basis();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::face_geometry::WeightedLandmarkRef >&
      procrustes_landmark_basis() const;

  // optional .mediapipe.face_geometry.Mesh3d canonical_mesh = 1;
  bool has_canonical_mesh() const;
  private:
  bool _internal_has_canonical_mesh() const;
  public:
  void clear_canonical_mesh();
  const ::mediapipe::face_geometry::Mesh3d& canonical_mesh() const;
  ::mediapipe::face_geometry::Mesh3d* release_canonical_mesh();
  ::mediapipe::face_geometry::Mesh3d* mutable_canonical_mesh();
  void set_allocated_canonical_mesh(::mediapipe::face_geometry::Mesh3d* canonical_mesh);
  private:
  const ::mediapipe::face_geometry::Mesh3d& _internal_canonical_mesh() const;
  ::mediapipe::face_geometry::Mesh3d* _internal_mutable_canonical_mesh();
  public:

  // optional .mediapipe.face_geometry.InputSource input_source = 3;
  bool has_input_source() const;
  private:
  bool _internal_has_input_source() const;
  public:
  void clear_input_source();
  ::mediapipe::face_geometry::InputSource input_source() const;
  void set_input_source(::mediapipe::face_geometry::InputSource value);
  private:
  ::mediapipe::face_geometry::InputSource _internal_input_source() const;
  void _internal_set_input_source(::mediapipe::face_geometry::InputSource value);
  public:

  // @@protoc_insertion_point(class_scope:mediapipe.face_geometry.GeometryPipelineMetadata)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::face_geometry::WeightedLandmarkRef > procrustes_landmark_basis_;
  ::mediapipe::face_geometry::Mesh3d* canonical_mesh_;
  int input_source_;
  friend struct ::TableStruct_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fgeometry_5fpipeline_5fmetadata_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// WeightedLandmarkRef

// optional uint32 landmark_id = 1;
inline bool WeightedLandmarkRef::_internal_has_landmark_id() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool WeightedLandmarkRef::has_landmark_id() const {
  return _internal_has_landmark_id();
}
inline void WeightedLandmarkRef::clear_landmark_id() {
  landmark_id_ = 0u;
  _has_bits_[0] &= ~0x00000001u;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 WeightedLandmarkRef::_internal_landmark_id() const {
  return landmark_id_;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 WeightedLandmarkRef::landmark_id() const {
  // @@protoc_insertion_point(field_get:mediapipe.face_geometry.WeightedLandmarkRef.landmark_id)
  return _internal_landmark_id();
}
inline void WeightedLandmarkRef::_internal_set_landmark_id(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  _has_bits_[0] |= 0x00000001u;
  landmark_id_ = value;
}
inline void WeightedLandmarkRef::set_landmark_id(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  _internal_set_landmark_id(value);
  // @@protoc_insertion_point(field_set:mediapipe.face_geometry.WeightedLandmarkRef.landmark_id)
}

// optional float weight = 2;
inline bool WeightedLandmarkRef::_internal_has_weight() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool WeightedLandmarkRef::has_weight() const {
  return _internal_has_weight();
}
inline void WeightedLandmarkRef::clear_weight() {
  weight_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline float WeightedLandmarkRef::_internal_weight() const {
  return weight_;
}
inline float WeightedLandmarkRef::weight() const {
  // @@protoc_insertion_point(field_get:mediapipe.face_geometry.WeightedLandmarkRef.weight)
  return _internal_weight();
}
inline void WeightedLandmarkRef::_internal_set_weight(float value) {
  _has_bits_[0] |= 0x00000002u;
  weight_ = value;
}
inline void WeightedLandmarkRef::set_weight(float value) {
  _internal_set_weight(value);
  // @@protoc_insertion_point(field_set:mediapipe.face_geometry.WeightedLandmarkRef.weight)
}

// -------------------------------------------------------------------

// GeometryPipelineMetadata

// optional .mediapipe.face_geometry.InputSource input_source = 3;
inline bool GeometryPipelineMetadata::_internal_has_input_source() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool GeometryPipelineMetadata::has_input_source() const {
  return _internal_has_input_source();
}
inline void GeometryPipelineMetadata::clear_input_source() {
  input_source_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline ::mediapipe::face_geometry::InputSource GeometryPipelineMetadata::_internal_input_source() const {
  return static_cast< ::mediapipe::face_geometry::InputSource >(input_source_);
}
inline ::mediapipe::face_geometry::InputSource GeometryPipelineMetadata::input_source() const {
  // @@protoc_insertion_point(field_get:mediapipe.face_geometry.GeometryPipelineMetadata.input_source)
  return _internal_input_source();
}
inline void GeometryPipelineMetadata::_internal_set_input_source(::mediapipe::face_geometry::InputSource value) {
  assert(::mediapipe::face_geometry::InputSource_IsValid(value));
  _has_bits_[0] |= 0x00000002u;
  input_source_ = value;
}
inline void GeometryPipelineMetadata::set_input_source(::mediapipe::face_geometry::InputSource value) {
  _internal_set_input_source(value);
  // @@protoc_insertion_point(field_set:mediapipe.face_geometry.GeometryPipelineMetadata.input_source)
}

// optional .mediapipe.face_geometry.Mesh3d canonical_mesh = 1;
inline bool GeometryPipelineMetadata::_internal_has_canonical_mesh() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  PROTOBUF_ASSUME(!value || canonical_mesh_ != nullptr);
  return value;
}
inline bool GeometryPipelineMetadata::has_canonical_mesh() const {
  return _internal_has_canonical_mesh();
}
inline const ::mediapipe::face_geometry::Mesh3d& GeometryPipelineMetadata::_internal_canonical_mesh() const {
  const ::mediapipe::face_geometry::Mesh3d* p = canonical_mesh_;
  return p != nullptr ? *p : *reinterpret_cast<const ::mediapipe::face_geometry::Mesh3d*>(
      &::mediapipe::face_geometry::_Mesh3d_default_instance_);
}
inline const ::mediapipe::face_geometry::Mesh3d& GeometryPipelineMetadata::canonical_mesh() const {
  // @@protoc_insertion_point(field_get:mediapipe.face_geometry.GeometryPipelineMetadata.canonical_mesh)
  return _internal_canonical_mesh();
}
inline ::mediapipe::face_geometry::Mesh3d* GeometryPipelineMetadata::release_canonical_mesh() {
  // @@protoc_insertion_point(field_release:mediapipe.face_geometry.GeometryPipelineMetadata.canonical_mesh)
  _has_bits_[0] &= ~0x00000001u;
  ::mediapipe::face_geometry::Mesh3d* temp = canonical_mesh_;
  canonical_mesh_ = nullptr;
  return temp;
}
inline ::mediapipe::face_geometry::Mesh3d* GeometryPipelineMetadata::_internal_mutable_canonical_mesh() {
  _has_bits_[0] |= 0x00000001u;
  if (canonical_mesh_ == nullptr) {
    auto* p = CreateMaybeMessage<::mediapipe::face_geometry::Mesh3d>(GetArenaNoVirtual());
    canonical_mesh_ = p;
  }
  return canonical_mesh_;
}
inline ::mediapipe::face_geometry::Mesh3d* GeometryPipelineMetadata::mutable_canonical_mesh() {
  // @@protoc_insertion_point(field_mutable:mediapipe.face_geometry.GeometryPipelineMetadata.canonical_mesh)
  return _internal_mutable_canonical_mesh();
}
inline void GeometryPipelineMetadata::set_allocated_canonical_mesh(::mediapipe::face_geometry::Mesh3d* canonical_mesh) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(canonical_mesh_);
  }
  if (canonical_mesh) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena = nullptr;
    if (message_arena != submessage_arena) {
      canonical_mesh = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, canonical_mesh, submessage_arena);
    }
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  canonical_mesh_ = canonical_mesh;
  // @@protoc_insertion_point(field_set_allocated:mediapipe.face_geometry.GeometryPipelineMetadata.canonical_mesh)
}

// repeated .mediapipe.face_geometry.WeightedLandmarkRef procrustes_landmark_basis = 2;
inline int GeometryPipelineMetadata::_internal_procrustes_landmark_basis_size() const {
  return procrustes_landmark_basis_.size();
}
inline int GeometryPipelineMetadata::procrustes_landmark_basis_size() const {
  return _internal_procrustes_landmark_basis_size();
}
inline void GeometryPipelineMetadata::clear_procrustes_landmark_basis() {
  procrustes_landmark_basis_.Clear();
}
inline ::mediapipe::face_geometry::WeightedLandmarkRef* GeometryPipelineMetadata::mutable_procrustes_landmark_basis(int index) {
  // @@protoc_insertion_point(field_mutable:mediapipe.face_geometry.GeometryPipelineMetadata.procrustes_landmark_basis)
  return procrustes_landmark_basis_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::face_geometry::WeightedLandmarkRef >*
GeometryPipelineMetadata::mutable_procrustes_landmark_basis() {
  // @@protoc_insertion_point(field_mutable_list:mediapipe.face_geometry.GeometryPipelineMetadata.procrustes_landmark_basis)
  return &procrustes_landmark_basis_;
}
inline const ::mediapipe::face_geometry::WeightedLandmarkRef& GeometryPipelineMetadata::_internal_procrustes_landmark_basis(int index) const {
  return procrustes_landmark_basis_.Get(index);
}
inline const ::mediapipe::face_geometry::WeightedLandmarkRef& GeometryPipelineMetadata::procrustes_landmark_basis(int index) const {
  // @@protoc_insertion_point(field_get:mediapipe.face_geometry.GeometryPipelineMetadata.procrustes_landmark_basis)
  return _internal_procrustes_landmark_basis(index);
}
inline ::mediapipe::face_geometry::WeightedLandmarkRef* GeometryPipelineMetadata::_internal_add_procrustes_landmark_basis() {
  return procrustes_landmark_basis_.Add();
}
inline ::mediapipe::face_geometry::WeightedLandmarkRef* GeometryPipelineMetadata::add_procrustes_landmark_basis() {
  // @@protoc_insertion_point(field_add:mediapipe.face_geometry.GeometryPipelineMetadata.procrustes_landmark_basis)
  return _internal_add_procrustes_landmark_basis();
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::mediapipe::face_geometry::WeightedLandmarkRef >&
GeometryPipelineMetadata::procrustes_landmark_basis() const {
  // @@protoc_insertion_point(field_list:mediapipe.face_geometry.GeometryPipelineMetadata.procrustes_landmark_basis)
  return procrustes_landmark_basis_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace face_geometry
}  // namespace mediapipe

PROTOBUF_NAMESPACE_OPEN

template <> struct is_proto_enum< ::mediapipe::face_geometry::InputSource> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::mediapipe::face_geometry::InputSource>() {
  return ::mediapipe::face_geometry::InputSource_descriptor();
}

PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_mediapipe_2fmodules_2fface_5fgeometry_2fprotos_2fgeometry_5fpipeline_5fmetadata_2eproto