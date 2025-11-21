// package: neuracore_types
// file: neuracore_types.proto

import * as jspb from "google-protobuf";

export class FloatRow extends jspb.Message {
  clearValuesList(): void;
  getValuesList(): Array<number>;
  setValuesList(value: Array<number>): void;
  addValues(value: number, index?: number): number;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): FloatRow.AsObject;
  static toObject(includeInstance: boolean, msg: FloatRow): FloatRow.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: FloatRow, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): FloatRow;
  static deserializeBinaryFromReader(message: FloatRow, reader: jspb.BinaryReader): FloatRow;
}

export namespace FloatRow {
  export type AsObject = {
    valuesList: Array<number>,
  }
}

export class JointData extends jspb.Message {
  getTimestamp(): number;
  setTimestamp(value: number): void;

  getValuesMap(): jspb.Map<string, number>;
  clearValuesMap(): void;
  getAdditionalValuesMap(): jspb.Map<string, number>;
  clearAdditionalValuesMap(): void;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): JointData.AsObject;
  static toObject(includeInstance: boolean, msg: JointData): JointData.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: JointData, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): JointData;
  static deserializeBinaryFromReader(message: JointData, reader: jspb.BinaryReader): JointData;
}

export namespace JointData {
  export type AsObject = {
    timestamp: number,
    valuesMap: Array<[string, number]>,
    additionalValuesMap: Array<[string, number]>,
  }
}

export class CameraData extends jspb.Message {
  getTimestamp(): number;
  setTimestamp(value: number): void;

  getFrameIdx(): number;
  setFrameIdx(value: number): void;

  clearExtrinsicsList(): void;
  getExtrinsicsList(): Array<FloatRow>;
  setExtrinsicsList(value: Array<FloatRow>): void;
  addExtrinsics(value?: FloatRow, index?: number): FloatRow;

  clearIntrinsicsList(): void;
  getIntrinsicsList(): Array<FloatRow>;
  setIntrinsicsList(value: Array<FloatRow>): void;
  addIntrinsics(value?: FloatRow, index?: number): FloatRow;

  getFrame(): Uint8Array | string;
  getFrame_asU8(): Uint8Array;
  getFrame_asB64(): string;
  setFrame(value: Uint8Array | string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): CameraData.AsObject;
  static toObject(includeInstance: boolean, msg: CameraData): CameraData.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: CameraData, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): CameraData;
  static deserializeBinaryFromReader(message: CameraData, reader: jspb.BinaryReader): CameraData;
}

export namespace CameraData {
  export type AsObject = {
    timestamp: number,
    frameIdx: number,
    extrinsicsList: Array<FloatRow.AsObject>,
    intrinsicsList: Array<FloatRow.AsObject>,
    frame: Uint8Array | string,
  }
}

export class PoseData extends jspb.Message {
  getTimestamp(): number;
  setTimestamp(value: number): void;

  getPoseMap(): jspb.Map<string, FloatRow>;
  clearPoseMap(): void;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): PoseData.AsObject;
  static toObject(includeInstance: boolean, msg: PoseData): PoseData.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: PoseData, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): PoseData;
  static deserializeBinaryFromReader(message: PoseData, reader: jspb.BinaryReader): PoseData;
}

export namespace PoseData {
  export type AsObject = {
    timestamp: number,
    poseMap: Array<[string, FloatRow.AsObject]>,
  }
}

export class EndEffectorData extends jspb.Message {
  getTimestamp(): number;
  setTimestamp(value: number): void;

  getOpenAmountsMap(): jspb.Map<string, number>;
  clearOpenAmountsMap(): void;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): EndEffectorData.AsObject;
  static toObject(includeInstance: boolean, msg: EndEffectorData): EndEffectorData.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: EndEffectorData, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): EndEffectorData;
  static deserializeBinaryFromReader(message: EndEffectorData, reader: jspb.BinaryReader): EndEffectorData;
}

export namespace EndEffectorData {
  export type AsObject = {
    timestamp: number,
    openAmountsMap: Array<[string, number]>,
  }
}

export class EndEffectorPoseData extends jspb.Message {
  getTimestamp(): number;
  setTimestamp(value: number): void;

  getPosesMap(): jspb.Map<string, FloatRow>;
  clearPosesMap(): void;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): EndEffectorPoseData.AsObject;
  static toObject(includeInstance: boolean, msg: EndEffectorPoseData): EndEffectorPoseData.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: EndEffectorPoseData, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): EndEffectorPoseData;
  static deserializeBinaryFromReader(message: EndEffectorPoseData, reader: jspb.BinaryReader): EndEffectorPoseData;
}

export namespace EndEffectorPoseData {
  export type AsObject = {
    timestamp: number,
    posesMap: Array<[string, FloatRow.AsObject]>,
  }
}

export class ParallelGripperOpenAmountData extends jspb.Message {
  getTimestamp(): number;
  setTimestamp(value: number): void;

  getOpenAmountsMap(): jspb.Map<string, number>;
  clearOpenAmountsMap(): void;
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ParallelGripperOpenAmountData.AsObject;
  static toObject(includeInstance: boolean, msg: ParallelGripperOpenAmountData): ParallelGripperOpenAmountData.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ParallelGripperOpenAmountData, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ParallelGripperOpenAmountData;
  static deserializeBinaryFromReader(message: ParallelGripperOpenAmountData, reader: jspb.BinaryReader): ParallelGripperOpenAmountData;
}

export namespace ParallelGripperOpenAmountData {
  export type AsObject = {
    timestamp: number,
    openAmountsMap: Array<[string, number]>,
  }
}

export class PointCloudData extends jspb.Message {
  getTimestamp(): number;
  setTimestamp(value: number): void;

  getPoints(): Uint8Array | string;
  getPoints_asU8(): Uint8Array;
  getPoints_asB64(): string;
  setPoints(value: Uint8Array | string): void;

  getRgbPoints(): Uint8Array | string;
  getRgbPoints_asU8(): Uint8Array;
  getRgbPoints_asB64(): string;
  setRgbPoints(value: Uint8Array | string): void;

  getExtrinsics(): Uint8Array | string;
  getExtrinsics_asU8(): Uint8Array;
  getExtrinsics_asB64(): string;
  setExtrinsics(value: Uint8Array | string): void;

  getIntrinsics(): Uint8Array | string;
  getIntrinsics_asU8(): Uint8Array;
  getIntrinsics_asB64(): string;
  setIntrinsics(value: Uint8Array | string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): PointCloudData.AsObject;
  static toObject(includeInstance: boolean, msg: PointCloudData): PointCloudData.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: PointCloudData, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): PointCloudData;
  static deserializeBinaryFromReader(message: PointCloudData, reader: jspb.BinaryReader): PointCloudData;
}

export namespace PointCloudData {
  export type AsObject = {
    timestamp: number,
    points: Uint8Array | string,
    rgbPoints: Uint8Array | string,
    extrinsics: Uint8Array | string,
    intrinsics: Uint8Array | string,
  }
}

export class LanguageData extends jspb.Message {
  getTimestamp(): number;
  setTimestamp(value: number): void;

  getText(): string;
  setText(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): LanguageData.AsObject;
  static toObject(includeInstance: boolean, msg: LanguageData): LanguageData.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: LanguageData, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): LanguageData;
  static deserializeBinaryFromReader(message: LanguageData, reader: jspb.BinaryReader): LanguageData;
}

export namespace LanguageData {
  export type AsObject = {
    timestamp: number,
    text: string,
  }
}

export class CustomData extends jspb.Message {
  getTimestamp(): number;
  setTimestamp(value: number): void;

  getData(): Uint8Array | string;
  getData_asU8(): Uint8Array;
  getData_asB64(): string;
  setData(value: Uint8Array | string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): CustomData.AsObject;
  static toObject(includeInstance: boolean, msg: CustomData): CustomData.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: CustomData, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): CustomData;
  static deserializeBinaryFromReader(message: CustomData, reader: jspb.BinaryReader): CustomData;
}

export namespace CustomData {
  export type AsObject = {
    timestamp: number,
    data: Uint8Array | string,
  }
}

