// Auto-generated. Do not edit!

// (in-package scara_command.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class SetCartesianVelRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.Vx = null;
      this.Vy = null;
      this.Vz = null;
      this.Wx = null;
      this.Wy = null;
      this.Wz = null;
    }
    else {
      if (initObj.hasOwnProperty('Vx')) {
        this.Vx = initObj.Vx
      }
      else {
        this.Vx = 0.0;
      }
      if (initObj.hasOwnProperty('Vy')) {
        this.Vy = initObj.Vy
      }
      else {
        this.Vy = 0.0;
      }
      if (initObj.hasOwnProperty('Vz')) {
        this.Vz = initObj.Vz
      }
      else {
        this.Vz = 0.0;
      }
      if (initObj.hasOwnProperty('Wx')) {
        this.Wx = initObj.Wx
      }
      else {
        this.Wx = 0.0;
      }
      if (initObj.hasOwnProperty('Wy')) {
        this.Wy = initObj.Wy
      }
      else {
        this.Wy = 0.0;
      }
      if (initObj.hasOwnProperty('Wz')) {
        this.Wz = initObj.Wz
      }
      else {
        this.Wz = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetCartesianVelRequest
    // Serialize message field [Vx]
    bufferOffset = _serializer.float32(obj.Vx, buffer, bufferOffset);
    // Serialize message field [Vy]
    bufferOffset = _serializer.float32(obj.Vy, buffer, bufferOffset);
    // Serialize message field [Vz]
    bufferOffset = _serializer.float32(obj.Vz, buffer, bufferOffset);
    // Serialize message field [Wx]
    bufferOffset = _serializer.float32(obj.Wx, buffer, bufferOffset);
    // Serialize message field [Wy]
    bufferOffset = _serializer.float32(obj.Wy, buffer, bufferOffset);
    // Serialize message field [Wz]
    bufferOffset = _serializer.float32(obj.Wz, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetCartesianVelRequest
    let len;
    let data = new SetCartesianVelRequest(null);
    // Deserialize message field [Vx]
    data.Vx = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [Vy]
    data.Vy = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [Vz]
    data.Vz = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [Wx]
    data.Wx = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [Wy]
    data.Wy = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [Wz]
    data.Wz = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 24;
  }

  static datatype() {
    // Returns string type for a service object
    return 'scara_command/SetCartesianVelRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '9ab779f0d7049dba7eeb7834b2b71119';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 Vx
    float32 Vy
    float32 Vz
    float32 Wx
    float32 Wy
    float32 Wz
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetCartesianVelRequest(null);
    if (msg.Vx !== undefined) {
      resolved.Vx = msg.Vx;
    }
    else {
      resolved.Vx = 0.0
    }

    if (msg.Vy !== undefined) {
      resolved.Vy = msg.Vy;
    }
    else {
      resolved.Vy = 0.0
    }

    if (msg.Vz !== undefined) {
      resolved.Vz = msg.Vz;
    }
    else {
      resolved.Vz = 0.0
    }

    if (msg.Wx !== undefined) {
      resolved.Wx = msg.Wx;
    }
    else {
      resolved.Wx = 0.0
    }

    if (msg.Wy !== undefined) {
      resolved.Wy = msg.Wy;
    }
    else {
      resolved.Wy = 0.0
    }

    if (msg.Wz !== undefined) {
      resolved.Wz = msg.Wz;
    }
    else {
      resolved.Wz = 0.0
    }

    return resolved;
    }
};

class SetCartesianVelResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.success = null;
    }
    else {
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetCartesianVelResponse
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetCartesianVelResponse
    let len;
    let data = new SetCartesianVelResponse(null);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 1;
  }

  static datatype() {
    // Returns string type for a service object
    return 'scara_command/SetCartesianVelResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '358e233cde0c8a8bcfea4ce193f8fc15';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    bool success
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetCartesianVelResponse(null);
    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    return resolved;
    }
};

module.exports = {
  Request: SetCartesianVelRequest,
  Response: SetCartesianVelResponse,
  md5sum() { return '283707b7ad403b8c31a22ddc99890608'; },
  datatype() { return 'scara_command/SetCartesianVel'; }
};
